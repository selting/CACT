from datetime import timedelta
from typing import Sequence, Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundling_and_bidding.fitness_functions.bundle_fitness import BundleFitnessFunction
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.optimization_policy import OptimizationPolicy
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.target_function import TargetFunction
from auction_module.bundling_and_bidding.type_defs import QueriesType, ResponsesType
from core_module.carrier import Carrier
from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request
from routing_module.routing_solver import RoutingSolver
from utility_module.utils import ACCEPTANCE_START_TIME, EXECUTION_START_TIME, END_TIME


class _CarrierModel:
    def __init__(self,
                 carrier_idx: int,
                 label: str,
                 depot: Depot,
                 num_unknown_orders: int,
                 num_vehicles: int,
                 routing_solver: RoutingSolver,
                 ):
        self._carrier_idx = carrier_idx
        self._label = label
        self._depot = depot
        self._num_unknown_orders = num_unknown_orders
        self._num_vehicles = num_vehicles

        self._routing_solver = routing_solver

        self.params_names: tuple[str] = tuple([f'x{i}' for i in range(self._num_unknown_orders)] +
                                              [f'y{i}' for i in range(self._num_unknown_orders)])
        self._current_params: dict[str, Optional[float]] = {k: None for k in self.params_names}
        self._current_params_have_changed = True
        self._requests = None
        self._cost_without_bundle = None

        self._current_model_without_bundle = None

    @property
    def current_params(self):
        return self._current_params

    @current_params.setter
    def current_params(self, value):
        self._current_params = value
        self._current_params_have_changed = True

    def get_coords(self) -> list[tuple[float, float]]:
        x_coords = [self.current_params[f'x{i}'] for i in range(self._num_unknown_orders)]
        y_coords = [self.current_params[f'y{i}'] for i in range(self._num_unknown_orders)]
        return list(zip(x_coords, y_coords))

    def requests(self, starting_uid: int, starting_index: int):
        """
        returns a Request for each of the x,y pairs of the self.current_params.
        It is important that these new, fictitious requests have the correct uids for indexing the distance and duration
        matrices. The starting uid is usually instance.num_vertices.

        :param starting_uid: instance.num_vertices
        :param starting_index: instance.num_requests
        :return:
        """
        if self._current_params_have_changed or not self._requests:
            own_requests = []
            for i, (x, y) in enumerate(self.get_coords()):
                request = Request(
                    vertex_uid=starting_uid + i,
                    label='est_' + str(starting_index + i),
                    index=starting_index + i,
                    x=x,
                    y=y,
                    initial_carrier_assignment=self._carrier_idx,
                    disclosure_time=ACCEPTANCE_START_TIME,
                    revenue=1,
                    load=1,
                    service_duration=timedelta(minutes=4),  # TODO these values are rather arbitrary
                    tw_open=EXECUTION_START_TIME,
                    tw_close=END_TIME)
                own_requests.append(request)
            self._requests = own_requests
        return self._requests

    def _current_params_instance(self, original_instance: CAHDInstance) -> CAHDInstance:
        """
        generate a new instance that includes the fictitious requests as defined by the current_params of the model
        :param original_instance:
        :return:
        """
        own_requests = self.requests(original_instance.num_vertices, original_instance.num_requests)
        extended_requests = original_instance.requests + own_requests
        extended_xy = np.array([(v.x, v.y) for v in original_instance.vertices + own_requests])
        if original_instance.meta['type'] == 'euclidean':
            extended_distance_matrix = squareform(pdist(extended_xy, metric='euclidean'))
            # generate the duration matrix in datetime.timedelta format
            constant_kmh = 30
            extended_duration_matrix = extended_distance_matrix / constant_kmh
            extended_duration_matrix = np.vectorize(lambda x: timedelta(hours=x))(extended_duration_matrix)
        else:
            raise NotImplementedError('Not yet correctly implemented for vienna type instances. OSRM distances are not'
                                      'available for the fictitious customers')
            # FIXME: find better approximations of the true (OSRM) distance?
            distance_matrix = squareform(pdist(xy))
            # FIXME: find better approximations of the true (OSRM) duration? For now, this at least works:
            # compute a scaling factor for the duration matrix by dividing the travel_duration_matrix by a computed
            # Euclidean distance matrix of the real instance and taking the mean of all those entries:
            inst_xy = np.array([(v.x, v.y) for v in original_instance.vertices])
            inst_eucl_dist = squareform(pdist(inst_xy, metric='euclidean'))
            inst_travel_duration = total_seconds_vectorized(original_instance._travel_duration_matrix)
            dist_scaling_matrix = np.divide(
                inst_travel_duration, inst_eucl_dist, out=np.zeros_like(inst_travel_duration),
                where=inst_eucl_dist != 0)
            dist_scaling_factor = np.nanmean(dist_scaling_matrix)

            duration_matrix = distance_matrix * dist_scaling_factor
            duration_matrix = np.vectorize(lambda x: timedelta(seconds=x))(duration_matrix)

        current_params_instance = CAHDInstance(
            id_='t=tmp_dummy',
            meta=original_instance.meta,
            carriers_max_num_tours=original_instance.carriers_max_num_tours,
            max_vehicle_load=original_instance.max_vehicle_load,
            max_tour_distance=original_instance.max_tour_distance,
            max_tour_duration=original_instance.max_tour_duration,
            requests=extended_requests,
            depots=original_instance.depots,
            duration_matrix=extended_duration_matrix,
            distance_matrix=extended_distance_matrix,
        )
        return current_params_instance

    def _carrier_without_bundle(self, original_instance: CAHDInstance):
        """
        Create a carrier model based on the current parameters, and thus it includes only the "unknown" requests.
        :return:
        """
        carrier_without_bundle = Carrier(self._carrier_idx, f'model of carrier {self._carrier_idx}', self._depot,
                                         'duration', self._num_vehicles, self._routing_solver)
        for request in self.requests(original_instance.num_vertices, original_instance.num_requests):
            carrier_without_bundle.assign_request(request)
            carrier_without_bundle.accept_request(request)
        return carrier_without_bundle

    def compute_bid_on_bundles(self, original_instance: CAHDInstance, bundles: Sequence[Bundle]):
        """
        For each element in elements, predict the valuation of the bundle for the carrier. The valuation is the
        difference in cost between the carrier's route with and without the bundle, i.e. the insertion cost.
        :param bundles:
        :return:
        """
        current_params_instance = self._current_params_instance(original_instance)  # instance + current_params
        carrier_without_bundle = self._carrier_without_bundle(original_instance)
        # TODO this should not necessarily be static
        carrier_without_bundle.route_all_accepted_statically(current_params_instance)

        predictions = []
        for query in bundles:
            bid = carrier_without_bundle.compute_bid_on_bundle(current_params_instance, query)
            predictions.append(bid)
        # TODO what about 'infeasible' bids?
        return np.array(predictions, dtype=float)

    def plot(self, ax: plt.Axes | None = None):

        if ax is None:
            fig, ax = plt.subplots()
        kwargs = dict(c="tab:red", marker="*", zorder=3, s=50, linestyle='--')
        ax.scatter(
            self._depot.x,
            self._depot.y,
            label=f'CarrierModel depot',
            **kwargs,
        )
        ax.scatter(*zip(*self.get_coords()),
                   s=40,
                   edgecolors='black',
                   facecolors='none',
                   # alpha=0.5,
                   linestyle='--',
                   label='CarrierModel current_params')
        # ax.grid(color="grey", linestyle="solid", linewidth=0.2)
        # ax.set_aspect("equal", "datalim")
        # ax.legend(frameon=True)
        # if axis_lims == 'euclidean':
        #     ax1.set_xlim(0, 25)
        #     ax1.set_ylim(0, 25)
        # elif axis_lims == 'vienna':
        #     raise NotImplementedError()
        # elif axis_lims is None:
        #     pass
        # else:
        #     raise ValueError
        return ax


class BundleFitnessCarrierModel(BundleFitnessFunction):
    def __init__(self, higher_is_better: bool, optimization_policy: OptimizationPolicy, error_function,
                 num_unknown_orders: int, num_vehicles: int, routing_solver: RoutingSolver,
                 prediction_metrics: list | None = None):
        """

        :param higher_is_better:
        :param optimization_policy:
        :param error_function: the function to minimize. must accept two arguments y and y_pred. E.g. MSE
        :param num_unknown_orders: the number of unknown orders to be predicted. This is the number of x,y pairs in the model.
        :param num_vehicles:
        :param routing_solver:
        :param prediction_metrics: a list of functions that accept two arguments y and y_pred. E.g. R2, MAE, MSE. These
        are used to evaluate the model's performance on the training data but are not used for optimization.
        """
        super().__init__()
        self._models: list[_CarrierModel] = []
        self._higher_is_better = 1 if higher_is_better else -1
        self._optimization_policy = optimization_policy
        self._error_function = error_function
        self._num_unknown_orders = num_unknown_orders
        self._num_vehicles = num_vehicles
        self._routing_solver = routing_solver
        self._prediction_metrics = [] if prediction_metrics is None else prediction_metrics
        self._params = {
            'optimization_policy': optimization_policy,
            'error_function': error_function.__name__,
            'num_unknown_orders': num_unknown_orders,
            'num_vehicles': num_vehicles,
            'routing_solver': routing_solver,
        }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.params})'

    def __call__(self, instance, bundles: Sequence[Bundle], **kwargs):
        bidder_idx = kwargs['bidder_idx']
        bid = self._models[bidder_idx].compute_bid_on_bundles(instance, bundles)
        return self._higher_is_better * bid

    def fit(self,
            instance: CAHDInstance,
            auction_request_pool: tuple[Request],
            queries: QueriesType,
            responses: ResponsesType):
        fit_results = []
        if not self._models:
            self._models = [
                _CarrierModel(
                    carrier_idx=idx,
                    label=str(idx),
                    depot=instance.depots[idx],
                    num_unknown_orders=self._num_unknown_orders,
                    num_vehicles=self._num_vehicles,
                    routing_solver=self._routing_solver
                ) for idx in range(len(queries))]

        for carrier_idx in range(len(self._models)):
            carrier_model = self._models[carrier_idx]
            target_func = TargetFunction(
                error_func=self._error_function,
                carrier_model=carrier_model,
                X=queries[carrier_idx],
                y_true=responses[carrier_idx],
                direction='min',
                target_func_pnames=carrier_model.params_names,
                target_func_pbounds=instance.meta['type'])
            target_opt, target_opt_params = self._optimization_policy.optimize(instance, auction_request_pool,
                                                                               target_func)
            self._models[carrier_idx].current_params = target_opt_params

            # compute the metrics
            y_true = responses[carrier_idx]
            y_pred = carrier_model.compute_bid_on_bundles(instance, queries[carrier_idx])  # costly ...
            fit_metrics_carrier = dict()
            for metric in self._prediction_metrics:
                metric_value = metric(y_true, y_pred)
                fit_metrics_carrier[metric.__name__] = metric_value
            fit_metrics_carrier['target_opt'] = target_opt
            fit_results.append(fit_metrics_carrier)
        return fit_results
