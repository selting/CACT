from abc import abstractmethod
from typing import Sequence

import mlflow
import numpy as np
import torch
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.optim import Adam
from torch.utils.data import DataLoader

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundling_and_bidding.fitness_functions.fitness_functions import FitnessFunction, device
from auction_module.bundling_and_bidding.type_defs import ResponsesType, QueriesType
from core_module.instance import CAHDInstance
from core_module.request import Request


class BundleFitnessFunction(FitnessFunction):
    @abstractmethod
    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        pass


class BundleFitnessFeature(BundleFitnessFunction):
    def __init__(self, bundle_feature, higher_is_better):
        super().__init__()
        self.bundle_feature = bundle_feature
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        return f'{prefix}BundleFeature({self.bundle_feature})'

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        if isinstance(bundles[0], int):  # if it's a single bundle
            NotImplementedError("This method is deprecated. Pass the complete population as numpy array instead.")
        fitness = []
        for bundle in bundles:
            if self.bundle_feature.startswith('carrier_specific'):
                value = bundle.get_feature(self.bundle_feature + f'_{kwargs["bidder_idx"]}')
            else:
                value = bundle.get_feature(self.bundle_feature)
            fitness.append(self.optimization_direction * value)
        return fitness


class BundleFitnessLinearRegression(BundleFitnessFunction):
    def __init__(self, interaction_degree: int = 1, higher_is_better: bool = True, metrics: None | list = None,
                 **model_hparams):
        """
        :param instance:
        :param auction_request_pool:
        :param alpha: Constant that multiplies the L1 term. Defaults to 1.0.
        :param interaction_degree: if interaction of components of the feature vector should be considered, set this
        parameter to the desired degree. For example, if the feature vector is [a, b], then the new vector for degree
        2 is [a, b, ab]
        :param higher_is_better:
        """
        super().__init__()
        self._models = []
        self._higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1
        self._interaction_degree = interaction_degree
        self._metrics = [] if metrics is None else metrics
        self._model_class = LinearRegression
        self._model_hparams = model_hparams

        self._params = {
            'interaction_degree': interaction_degree,
            'higher_is_better': higher_is_better,
            **model_hparams
        }

    def __repr__(self):
        prefix = '' if self._higher_is_better else '-'
        s = f'{prefix}{self._model_class.__name__}(interaction_degree={self._interaction_degree})'
        if self._model_hparams:
            s += str(self._model_hparams)
        return s

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        X = np.array([b.bitstring for b in bundles])
        if self._interaction_degree > 1:
            X = PolynomialFeatures(degree=self._interaction_degree,
                                   interaction_only=True,
                                   include_bias=False).fit_transform(X)
        return self.optimization_direction * self._models[bidder_idx].predict(X)

    def fit(self,
            instance: CAHDInstance,
            auction_request_pool: tuple[Request],
            queries: QueriesType,
            responses: ResponsesType):
        if not self._models:
            self._models = [self._model_class(**self._model_hparams) for _ in range(len(queries))]

        fit_metrics = []
        for bidder_idx in range(len(self._models)):
            X = queries[bidder_idx]
            X = np.array([bundle.bitstring for bundle in X])
            if self._interaction_degree > 1:
                X = PolynomialFeatures(degree=self._interaction_degree, interaction_only=True,
                                       include_bias=False).fit_transform(X)
            y_true = np.array(responses[bidder_idx])
            model = self._models[bidder_idx]
            model.fit(X, y_true)

            # compute and log loss and metrics
            y_pred = model.predict(X)
            fit_metrics_carrier = dict()
            parent_group_id = mlflow.active_run().data.tags['group_id']
            with mlflow.start_run(nested=True, run_name=f'Carrier {bidder_idx} - BundleFitness.fit',
                                  tags={'group_id': parent_group_id}):
                mlflow.log_params(self.params)
                mlflow.log_param('Carrier', bidder_idx)
                mlflow.log_param('num_samples', len(X))
                for metric in self._metrics:
                    metric_value = metric(y_pred, y_true)
                    mlflow.log_metric(metric.__name__, metric_value)
                    fit_metrics_carrier['bundle_fitness_' + metric.__name__] = metric_value

                # fig, ax = plt.subplots()
                # ax.plot(y_true, y_pred, 'o')
                # ax.axline((0, 0), slope=1, color='green', label='x=y', linestyle='--')
                # ax.set_xlabel('True Values')
                # ax.set_ylabel('Predictions')
                # ax.set_title('True Values vs Predictions')
                # mlflow.log_figure(fig)
                # plt.close(fig)
            fit_metrics.append(fit_metrics_carrier)
        return fit_metrics


class BundleFitnessLassoRegression(BundleFitnessLinearRegression):
    def __init__(self, interaction_degree: int = 1, higher_is_better: bool = True, metrics: None | list = None,
                 alpha: float = 1.0):
        """
        :param alpha: Constant that multiplies the L1 term. Defaults to 1.0.
        :param interaction_degree: if interaction of components of the feature vector should be considered, set this
        parameter to the desired degree. For example, if the feature vector is [a, b], then the new vector for degree
        2 is [a, b, ab]
        :param higher_is_better:
        """
        super().__init__(interaction_degree, higher_is_better, metrics, alpha=alpha)
        self._model_class = Lasso


class BundleFitnessRidgeRegression(BundleFitnessLinearRegression):
    def __init__(self, interaction_degree: int = 1, higher_is_better: bool = True, metrics: None | list = None,
                 **ridge_kwargs):
        """

        :param instance:
        :param auction_request_pool:
        :param alpha: regularization parameter
        :param interaction_degree: if interaction of components of the feature vector should be considered, set this
        parameter to the desired degree. For example, if the feature vector is [a, b], then the new vector for degree
        2 is [a, b, ab]
        :param higher_is_better:
        :param logging:
        """

        super().__init__(interaction_degree, higher_is_better, metrics, **ridge_kwargs)
        self._model_class = Ridge


class BundleFitnessElasticNet(BundleFitnessLinearRegression):
    def __init__(self,
                 interaction_degree: int = 1,
                 higher_is_better: bool = True,
                 metrics: list = None,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 ):
        """

        :param instance:
        :param auction_request_pool:
        :param alpha: regularization parameter
        :param interaction_degree: if interaction of components of the feature vector should be considered, set this
        parameter to the desired degree. For example, if the feature vector is [a, b], then the new vector for degree
        2 is [a, b, ab]
        :param higher_is_better:
        :param logging:
        """
        super().__init__(interaction_degree, higher_is_better, metrics, alpha=alpha, l1_ratio=l1_ratio)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._model_class = ElasticNet
        self._model_hparams = {'alpha': alpha, 'l1_ratio': l1_ratio}


class QueryResponseDataset(torch.utils.data.Dataset):
    def __init__(self, queries: Sequence[tuple[int]], responses: Sequence[float]):
        self.queries = queries
        self.responses = responses
        # normalize responses from (0, 8*60**2) to (0, 1)
        # self.responses = [r / (8 * 60 ** 2) for r in self.responses] # FIXME is this required? Always? Why (not)?
        pass

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        X = torch.tensor(self.queries[idx], device=device, dtype=torch.float32)
        y = torch.tensor(self.responses[idx], device=device, dtype=torch.float32).unsqueeze(-1)
        return X, y


class BundleFitnessNeuralNetwork(BundleFitnessFunction):
    def __init__(self,
                 instance: CAHDInstance,
                 auction_request_pool,
                 higher_is_better,
                 optimizer=Adam,
                 lr=0.001,
                 output_dim: int = 1,
                 hidden_layer_sizes=(100, 100, 100),
                 activation=torch.nn.ReLU,
                 loss_fn=torch.nn.MSELoss(reduction='mean'),
                 num_epochs=100,
                 batch_size=16,
                 # training_size=0.8,
                 logging=False
                 ):
        """

        :param instance:
        :param auction_request_pool:
        :param optimizer:
        :param lr:
        :param output_dim:
        :param hidden_layer_sizes:
        :param activation:
        :param reg:
        :param loss_fn:
        :param num_epochs:
        :param batch_size:
        :param higher_is_better:
        # :param training_size: how much of the data should be used for training, the rest is used for validation
        :param logging:
        """
        super().__init__()
        self.input_dim = len(auction_request_pool)
        self.output_dim = output_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.optimizer_type = optimizer
        self.lr = lr
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1
        # self.training_size = training_size

        self._initialize_models(instance)
        self.logging = logging

        self._params = {
            'optimizer': optimizer.__name__,
            'lr': lr,
            'output_dim': output_dim,
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation.__name__,
            'loss_fn': loss_fn.__class__.__name__,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'higher_is_better': higher_is_better,
            # 'training_size': training_size,
        }

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        name = f'NeuralNetwork({self.optimizer_type.__name__}, lr={self.lr}, ' \
               f'output_dim={self.output_dim}, hidden_layer_sizes={self.hidden_layer_sizes}, ' \
               f'activation={self.activation.__name__}, loss_fn={self.loss_fn}, ' \
               f'num_epochs={self.num_epochs}, batch_size={self.batch_size})'
        return prefix + name

    def _initialize_models(self, instance):
        # instantiate NN models
        self.models = []
        self.optimizers = []
        for _ in range(instance.num_carriers):
            layers = torch.nn.Sequential()

            in_dim = self.input_dim

            for i, out_dim in enumerate(self.hidden_layer_sizes):
                layers.append(torch.nn.Linear(in_dim, out_dim, bias=True, device=device, dtype=torch.float32))
                layers.append(self.activation())
                in_dim = out_dim

            # output layer
            layers.append(torch.nn.Linear(in_dim, self.output_dim, bias=True, device=device, dtype=torch.float32))
            # self.layers.append(self.activation()) ?????

            # use custom initialization ????
            # for layer in self.layers:
            #     if isinstance(layer, torch.nn.Linear):
            #         torch.nn.init.uniform_(layer.weight)
            #         torch.nn.init.constant_(layer.bias, 0)

            # optimizer
            optimizer = self.optimizer_type(layers.parameters(), lr=self.lr)

            self.models.append(layers)
            self.optimizers.append(optimizer)
            pass

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        # self.models[bidder_idx].eval()  # set model to evaluation mode ???
        with torch.no_grad():
            X = torch.tensor(bundles, device=device, dtype=torch.float32)
            y_hat = self.models[bidder_idx](X)
        return (self.optimization_direction * y_hat.squeeze()).tolist()

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        for bidder_idx in range(len(self.models)):
            b_model = self.models[bidder_idx]
            b_optimizer = self.optimizers[bidder_idx]
            b_queries = queries[bidder_idx]
            b_responses = responses[bidder_idx]

            dataset = QueryResponseDataset(b_queries, b_responses)
            # train_dataset, val_dataset = random_split(dataset, [self.training_size, 1 - self.training_size])
            # train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            # if val_dataset:
            #     val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
            #
            # for epoch in range(self.num_epochs):
            #     train_loss = self.train_loop(b_model, b_optimizer, train_dataloader)
            #     if val_dataset:
            #         val_loss = self.validation_loop(b_model, val_dataloader)
            #
            #     if self.logging:
            #         self.run[f'train_loss_{bidder_idx}'].append(train_loss.item())
            #         if val_dataset:
            #             self.run[f'val_loss_{bidder_idx}'].append(val_loss.item())

            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for epoch in range(self.num_epochs):
                train_loss = self.train_loop(b_model, b_optimizer, train_dataloader)
                if self.logging:
                    self.run[f'train_loss_{bidder_idx}'].append(train_loss.item())

        if self.logging:
            self.run.wait()
        pass

    def train_loop(self, b_model, b_optimizer, train_dataloader):
        # train
        b_model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            b_optimizer.zero_grad()
            # compute loss
            y_hat = b_model(X)
            loss = self.loss_fn(y_hat, y)
            train_loss += loss
            # backpropagation
            loss.backward()
            b_optimizer.step()
        train_loss /= len(train_dataloader)
        return train_loss

    def validation_loop(self, b_model, val_dataloader):
        # validate
        b_model.eval()  # set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                y_hat = b_model(X)
                val_loss += self.loss_fn(y_hat, y)
        val_loss /= len(val_dataloader)
        return val_loss


class BundleFitnessDecisionTreeRegressor(BundleFitnessFunction):
    """
    This fitness function uses a decision tree to predict the valuation of a bundle.
    It uses binary bundle representation.
    """

    def __init__(self,
                 instance: CAHDInstance,
                 auction_request_pool,
                 higher_is_better,
                 **kwargs):
        super().__init__()
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1
        self.models = [DecisionTreeRegressor(**kwargs) for _ in range(instance.num_carriers)]
        self._params = kwargs

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        hparams = ', '.join(f'{k}={v}' for k, v in self.params.items())
        hparams = '(' + hparams + ')' if hparams else ''
        return f'{prefix}DecisionTreeRegressor{hparams}'

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        if isinstance(bundles[0], int):  # if it's a single bundle
            fitness = self.optimization_direction * self.models[bidder_idx].predict([bundles])[0]
        else:  # if it's a numpy array of bundles
            fitness = self.optimization_direction * self.models[bidder_idx].predict(bundles)
        return fitness

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        for bidder_idx in range(len(self.models)):
            X = queries[bidder_idx]
            y = responses[bidder_idx]
            self.models[bidder_idx].fit(X, y)


class BundleFitnessRandomForestRegressor(BundleFitnessFunction):
    """
    This fitness function uses a Random Forest Regressor to predict the valuation of a bundle.
    It uses binary bundle representation.
    A good default value for n_estimators is 100-200
    A good default value for max_depth is None
    """

    def __init__(self,
                 instance: CAHDInstance,
                 auction_request_pool,
                 higher_is_better,
                 logging: bool = False,
                 metrics: list = (torch.nn.MSELoss(reduction='mean'), torch.nn.L1Loss(reduction='mean')),
                 **kwargs):
        super().__init__()
        self.models = [RandomForestRegressor(warm_start=False, n_jobs=1, **kwargs)
                       for _ in range(instance.num_carriers)]
        self.hparams = kwargs
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1
        self.metrics = metrics
        self.logging = logging

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        hparams = ', '.join(f'{k}={v}' for k, v in self.hparams.items())
        hparams = '(' + hparams + ')' if hparams else ''
        return f'{prefix}RandomForestRegressor{hparams}'

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        if isinstance(bundles[0], int):  # if it's a single bundle
            fitness = self.optimization_direction * self.models[bidder_idx].predict([bundles])[0]
        else:  # if it's a numpy array of bundles
            fitness = self.optimization_direction * self.models[bidder_idx].predict(bundles)
        return fitness

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        for bidder_idx in range(len(self.models)):
            X = queries[bidder_idx]
            y = responses[bidder_idx]
            model = self.models[bidder_idx]
            model.fit(X, y)

            # compute and log loss
            if self.logging:
                y_hat = model.predict(X)
                for m in self.metrics:
                    m_value = m(torch.tensor(y_hat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
                    self.run[f'{m.__class__.__name__}_{bidder_idx}'].append(m_value.item())
                fig, ax = plt.subplots()
                ax.plot(y, y_hat, 'o')
                ax.axline((0, 0), slope=1, color='green', label='x=y', linestyle='--')
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predictions')
                ax.set_title('True Values vs Predictions')
                self.run[f'scatter_plot_{bidder_idx}'].append(fig)
                plt.close(fig)


class BundleFitnessXGBoost(BundleFitnessFunction):
    """
    This fitness function uses XGBoost to predict the valuation of a bundle.
    It uses binary bundle representation.
    """

    def __init__(self,
                 instance: CAHDInstance,
                 auction_request_pool,
                 higher_is_better,
                 **kwargs):
        super().__init__()
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1

        self.models = [xgb.XGBRegressor(n_jobs=1, **kwargs) for _ in range(instance.num_carriers)]
        self._params = kwargs

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        params = ', '.join(f'{k}={v}' for k, v in self.params.items())
        params = '(' + params + ')' if params else ''
        return f'{prefix}XGBoost{params}'

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        if isinstance(bundles[0], int):  # if it's a single bundle
            fitness = self.optimization_direction * self.models[bidder_idx].predict([bundles])[0]
        else:  # if it's a numpy array of bundles
            fitness = self.optimization_direction * self.models[bidder_idx].predict(bundles)
        return fitness

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        for bidder_idx in range(len(self.models)):
            X = queries[bidder_idx]
            y = responses[bidder_idx]
            self.models[bidder_idx].fit(X, auction_request_pool, y, )


def exponential_kernel(X, Y):
    gamma = 1  # Replace with your desired gamma value
    return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y, axis=2))


class BundleFitnessSVR(BundleFitnessFunction):
    """
    This fitness function uses scikit learn SVR with a custom exponential kernel to predict the valuation of a bundle.
    With a high enough C and epsilon=0, SVR will fit the training data exactly, but it's not guaranteed to generalize
    It uses binary bundle representation.
    """

    def __init__(self, instance: CAHDInstance, auction_request_pool: tuple[Request], higher_is_better, **kwargs):
        super().__init__()
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1
        self._params = kwargs

        self.models = [SVR(kernel=exponential_kernel, C=kwargs.get('C', 1.0), epsilon=kwargs.get('epsilon', 0.1))
                       for _ in range(instance.num_carriers)]

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        hparams = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'{prefix}SVR{hparams}'

    def __call__(self, instance: CAHDInstance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        fitness = self.higher_is_better * self.models[bidder_idx].predict(bundles)
        return fitness

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        for bidder_idx in range(len(self.models)):
            X = queries[bidder_idx]
            y = responses[bidder_idx]
            self.models[bidder_idx].fit(X, y, verbose=False)
