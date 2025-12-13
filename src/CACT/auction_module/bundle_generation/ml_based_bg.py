import datetime as dt
import warnings
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Sequence, Optional, Set, Any, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import NeptuneLogger
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from auction_module.bundle_generation.SPO.cap_solver import CombinatorialAuctionSolver
from auction_module.bundle_generation.SPO.lightning_datamodule import custom_collate_fn
from auction_module.bundle_generation.SPO.models.lightning_module import LightningMLP
from auction_module.bundle_generation.SPO.torch_dataset import CAPDataset
from auction_module.bundle_generation.SPO.torch_losses import spo_loss
from auction_module.bundle_generation.SPO.transforms import ZeroOneScaler
from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.partition_based.partition import Partition
from auction_module.bundle_generation.partition_based.partition_fitness import DummyPartitionFitness
from core_module import instance as it, solution as slt
from core_module.request import Request
from utility_module.io import data_dir
from utility_module.utils import debugger_is_active

t_auction_bundle_pool = Sequence[Bundle]
t_auction_partition_pool = Optional[Sequence[Partition]]
t_BidsMatrix = dict[Bundle, dict[int, dt.timedelta]]


class MachineLearningBasedBundleGeneration(ABC):
    """
    Generates auction bundles based on partitioning the auction request pool. This guarantees a feasible solution
    for the Winner Determination Problem which cannot (easily) be guaranteed if bundles are generated without
    considering that the WDP requires a partitioning of the auction request pool (see bundle_based_bg.py)
    """

    def __init__(self):
        self.name = self.__class__.__name__

    def __repr__(self):
        return self.name

    def execute(self,
                instance: it.CAHDInstance,
                solution: slt.CAHDSolution,
                auction_request_pool: tuple[Request],
                num_bidding_jobs: int,
                original_assignment: Assignment
                ) -> tuple[t_auction_bundle_pool, t_auction_partition_pool]:
        auction_bundle_pool, auction_partition_pool = self.generate_auction_bundles(
            instance, solution,
            auction_request_pool,
            num_bidding_jobs,
            original_assignment)
        return tuple(auction_bundle_pool), auction_partition_pool

    @abstractmethod
    def generate_auction_bundles(
            self,
            instance: it.CAHDInstance,
            solution: slt.CAHDSolution,
            auction_request_pool: tuple[Request],
            num_bidding_jobs: int,
            original_assignment: Assignment
    ) -> tuple[t_auction_bundle_pool, t_auction_partition_pool]:
        pass


# class ActiveLearningCAPDataset(CAPDataset):
#     def __init__(self, X, y, solver):
#         super().__init__(X, y, solver)
#         self.labelled_indices = []  # should probably be a set
#         self.unlabelled_indices = list(range(len(self.X)))
#
#     def __len__(self):
#         return len(self.labelled_indices)
#
#     def __getitem__(self, idx):
#         return super().__getitem__(self.labelled_indices[idx])
#
#     def label(self, idx):
#         self.unlabelled_indices.remove(idx)
#         self.labelled_indices.append(idx)


class ActiveLearningBundleGeneration(MachineLearningBasedBundleGeneration):
    def __init__(self, bidding_behavior: BiddingBehavior, bundle_features: Sequence[str], loss_function=mse_loss,
                 metrics=None, max_epochs=500, learning_rate=0.001, num_training_partitions=16, num_val_partitions=16,
                 num_partitions_to_explore=200):
        super().__init__()
        if metrics is None:
            metrics = [spo_loss]
        self.bidding_behavior = bidding_behavior
        self.bundle_features = bundle_features
        self.partition_valuation = DummyPartitionFitness()
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.num_training_partitions = num_training_partitions
        self.num_val_partitions = num_val_partitions
        self.num_partitions_to_explore = num_partitions_to_explore
        self.name = 'ALBG' + '+' + self.bidding_behavior.name + '+' + self.loss_function.__name__ + \
                    '+' + str(len(self.bundle_features)) + 'features'

    def get_scaling_factors(self) -> pd.DataFrame:
        df = pd.read_excel(data_dir.joinpath('SPO/ActiveLearningScalingFactors.xlsx'), index_col=None)
        df = df.set_index(['n', 'o', 'v', 'time_window_length_hours', 'direction'])
        return df

    def generate_auction_bundles(
            self,
            instance: it.CAHDInstance,
            solution: slt.CAHDSolution,
            auction_request_pool: tuple[Request],
            num_bidding_jobs: int,
            original_assignment: Assignment
    ) -> tuple[set[Bundle], set[Union[Partition, Any]]]:

        module = LightningMLP(
            num_inputs=len(self.bundle_features),
            num_outputs=3,
            hidden_layers=0,
            hidden_units=0,
            activation_fn=None,
            loss_function=self.loss_function,
            learning_rate=self.learning_rate,
            l1_weight=0.0,
            metrics=self.metrics,
        )
        trainer = self._trainer(instance, solution)

        feature_scaler, target_scaler = self.get_scalers(instance, solution)

        train_partition_pool = {Partition.from_assignment(original_assignment),
                                *self.generate_random_partitions(instance, auction_request_pool,
                                                                 self.num_training_partitions,
                                                                 {Partition.from_assignment(original_assignment)})}
        train_dataloader = self._train_dataloader(instance, solution, train_partition_pool, auction_request_pool,
                                                  feature_scaler, target_scaler)

        val_dataloader = self._val_dataloader(instance, solution, train_partition_pool, auction_request_pool,
                                              feature_scaler, target_scaler)

        # train the model using the validation for monitoring the performance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        auction_bundle_pool: Set[Bundle] = {b for p in train_partition_pool for b in p.bundles}

        # generate new partitions using the trained model
        explore_partition_pool = self.generate_random_partitions(instance, auction_request_pool,
                                                                 self.num_partitions_to_explore,
                                                                 train_partition_pool)
        temp_dataset = self.CAPdataset_from_partition_pool_predicted(instance, auction_request_pool,
                                                                     explore_partition_pool, feature_scaler, module)
        zyp_tuples = [(z, y_pred, p) for z, y_pred, p in zip(temp_dataset.z, temp_dataset.y, explore_partition_pool)]
        zyp_sorted = sorted(zyp_tuples, key=itemgetter(0))  # minimum CAP objective value first

        # update auction_bundle_pool, auction_partition_pool [# and bids_matrix]
        # using sets to avoid duplicates
        auction_bundle_pool_final = set(original_assignment.bundles())
        auction_partition_pool_final = {Partition.from_assignment(original_assignment)}
        for z, y, p in zyp_sorted:
            auction_bundle_pool_final.update(p.bundles)
            auction_partition_pool_final.update({p})
            # y = target_scaler.inverse_transform(y)
            # for i in range(len(p.bundles)):
            #     bids_matrix[p.bundles[i]] = {j: dt.timedelta(seconds=y[i][j].item()) for j in range(len(y[i]))}
            if len(auction_bundle_pool_final) >= num_bidding_jobs:
                break

        return auction_bundle_pool_final, auction_partition_pool_final

    def _trainer(self, instance, solution):
        trainer = pl.Trainer(logger=NeptuneLogger(project='CR-AHD/ActiveLearning',
                                                  name='ActiveLearningBundleGeneration',
                                                  log_model_checkpoints=False, ),
                             callbacks=[EarlyStopping(monitor='early_stopping_monitor',
                                                      min_delta=0.001,
                                                      patience=50,
                                                      mode='min', verbose=True),
                                        # RichProgressBar(),
                                        ],
                             deterministic='warn',
                             max_epochs=self.max_epochs,
                             # min_epochs=100,
                             enable_checkpointing=False,
                             enable_progress_bar=False,
                             )
        logger = trainer.logger.experiment
        logger['training/data'] = {'bundle_features': str(self.bundle_features)}
        logger['training/instance'] = instance.meta
        logger['training/solver_config'] = self.parse_solver_config(solution.solver_config)
        return trainer

    def _train_dataloader(self, instance, solution, auction_partition_pool, auction_request_pool:tuple[Bundle], feature_scaler,
                          target_scaler):
        train_dataset, bids_matrix = self.CAPdataset_from_partition_pool(instance, solution, auction_request_pool,
                                                                         auction_partition_pool, feature_scaler,
                                                                         target_scaler)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=4,  # TODO make this a parameter
                                      shuffle=False,
                                      collate_fn=custom_collate_fn,
                                      num_workers=0, )
        return train_dataloader

    def _val_dataloader(self, instance, solution, auction_partition_pool, auction_request_pool:tuple[Bundle], feature_scaler,
                        target_scaler):
        # generate some random validation partitions
        val_partition_pool = self.generate_random_partitions(instance, auction_request_pool,
                                                             16,  # TODO make this a parameter
                                                             auction_partition_pool)
        val_dataset, _ = self.CAPdataset_from_partition_pool(instance, solution, auction_request_pool,
                                                             val_partition_pool, feature_scaler, target_scaler)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=4,  # TODO make this a parameter
                                    shuffle=False,
                                    collate_fn=custom_collate_fn,
                                    num_workers=0, )
        return val_dataloader

    def parse_solver_config(self, solver_config):
        parsed_solver_config = dict()
        for k, v in solver_config.items():
            if isinstance(v, dt.timedelta):
                parsed_solver_config[k] = v.total_seconds()
            elif isinstance(v, (float, int)):
                parsed_solver_config[k] = v
            else:
                parsed_solver_config[k] = str(v)
        return parsed_solver_config

    def generate_random_partitions(self, instance, request_pool, num_partitions,
                                   black_list: Optional[Set[Partition]] = None):
        """generates random partitions and returns them in a set. skips partitions in black_list"""
        if black_list is None:
            black_list = set()
        partitions = set()
        bundle_pool = []
        while len(partitions) < num_partitions:  # to avoid duplicates
            random_partition = Partition.random(instance, request_pool)
            random_partition.normalize(True)
            if (random_partition not in partitions) and (random_partition not in black_list):
                partitions.add(random_partition)
                bundle_pool.extend(random_partition.bundles)
        return tuple(partitions)  # return tuple to have a fixed order

    def get_scalers(self, instance, solution, feature=True, target=True):
        """returns ZeroOneScaler objects for features and targets"""
        feature_scaler = None
        target_scaler = None
        # scaling values are from other experiments, i.e. from historical data and may be slightly inaccurate
        # additionally, they highly depend on e.g. the overlap or time window length
        scaling_factors = self.get_scaling_factors()
        index = (
            instance.meta['n'],
            instance.meta['o'],
            instance.meta['v'],
            solution.solver_config['time_window_length'].total_seconds() / 60 ** 2,
        )

        if feature:
            feature_scaler = ZeroOneScaler(
                min_=torch.as_tensor(scaling_factors.loc[index + ('min',), self.bundle_features], dtype=torch.float32),
                max_=torch.as_tensor(scaling_factors.loc[index + ('max',), self.bundle_features], dtype=torch.float32),
                replace_fixed_value_features_with='original',
                across_features=False)
        if target:
            target_scaler = ZeroOneScaler(
                min_=torch.as_tensor(scaling_factors.loc[index + ('min',), 'bid'], dtype=torch.float32),
                max_=torch.as_tensor(scaling_factors.loc[index + ('max',), 'bid'], dtype=torch.float32),
                replace_fixed_value_features_with='original',
                across_features=True)
        return feature_scaler, target_scaler

    def CAPdataset_from_partition_pool(self, instance, solution, auction_request_pool:tuple[Bundle], partitions, feature_scaler,
                                       target_scaler) -> tuple[CAPDataset, pd.DataFrame]:
        X = []
        y_true = []
        bundles = [b for p in partitions for b in p.bundles]
        bids_matrix = pd.DataFrame(index=[b.requests for b in bundles],
                                   columns=[carrier.id_ for carrier in solution.carriers],
                                   )
        solver = []
        for partition in partitions:
            features = []
            for bundle in partition.bundles:
                f = torch.as_tensor([bundle.get_feature(feature) for feature in self.bundle_features],
                                    dtype=torch.float32)
                features.append(f)
            if feature_scaler:
                features = feature_scaler.transform(torch.stack(features))
            X.append(features)
            y_true_oracle: pd.DataFrame = self.query_oracle(instance, solution, partition.bundles)
            y_true_oracle_numpy = y_true_oracle.applymap(lambda x: x.total_seconds()).to_numpy(dtype=np.float32)
            y_true_tensor = torch.from_numpy(y_true_oracle_numpy)

            if target_scaler:
                y_true_tensor = target_scaler.transform(y_true_tensor)
            y_true.append(y_true_tensor)
            # bids_matrix.update(y_true_oracle, overwrite=False, errors='raise')
            # using .update() would have been the nicer approach but does not work because pandas auto-converts to
            #  pd.Timedelta but then messes up the converted values in the next iteration. This is a horrible hack:
            for bundle, row in zip(partition.bundles, y_true_oracle.to_numpy()):
                for col, bid in enumerate(row):
                    bids_matrix.loc[[bundle.requests, ], col] = bid
            solver_ = CombinatorialAuctionSolver(auction_request_pool, partition.bundles, instance.num_carriers)
            solver.append(solver_)
        dataset = CAPDataset(X, y_true, solver)  # solving will happen here
        return dataset, bids_matrix

    def CAPdataset_from_partition_pool_predicted(self, instance, auction_request_pool:tuple[Bundle], partitions,
                                                 feature_scaler, module):
        X = []
        # y_true = []
        # bids_matrix = dict()  # TODO bids_matrix class that has different representations (dict, tensor, df, numpy)?
        solver = []
        pbar = tqdm(total=sum(len(p.bundles) for p in partitions), desc='Computing bundle features', ncols=100,
                    disable=not debugger_is_active())
        for partition in partitions:
            features = []
            for bundle in partition.bundles:
                f = torch.as_tensor([bundle.get_feature(feature) for feature in self.bundle_features],
                                    dtype=torch.float32)
                features.append(f)
                pbar.update()

            features = torch.stack(features)
            if feature_scaler:
                features = feature_scaler.transform(features)
            X.append(features)
            # y_true_oracle: t_BidsMatrix = self.query_oracle(instance, solution, partition.bundles)
            # y_true_tensor = torch.as_tensor([[v_.total_seconds()
            #                                   for k_, v_ in v.items()]
            #                                  for k, v in y_true_oracle.items()])
            # if target_scaler:
            #     y_true_tensor = target_scaler.transform(y_true_tensor)
            # y_true.append(y_true_tensor)
            # bids_matrix.update(y_true_oracle)
            solver_ = CombinatorialAuctionSolver(auction_request_pool, partition.bundles, instance.num_carriers)
            solver.append(solver_)

        # TODO rather than doing this stuff manually, I could use the trainer.predict() with a new Dataset/Dataloader
        #  that does not require targets
        torch.set_grad_enabled(False)
        module.eval()
        y_pred = [module.predict_step(X[i], i) for i in range(len(X))]
        torch.set_grad_enabled(True)
        module.train()

        dataset = CAPDataset(X, y_pred, solver)  # solving will happen here
        return dataset

    def query_oracle(self, instance, solution, bundles) -> pd.DataFrame:
        y_true: pd.DataFrame = self.bidding_behavior.execute_bidding(instance, solution, bundles)
        return y_true
