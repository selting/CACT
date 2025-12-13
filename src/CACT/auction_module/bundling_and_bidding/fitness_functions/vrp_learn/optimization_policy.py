import math
import warnings
from abc import abstractmethod

from itertools import product

import mlflow
import nlopt
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound


from core_module.instance import CAHDInstance
from core_module.request import Request
from utility_module.parameterized_class import ParameterizedClass
from .target_function import TargetFunction, TargetFunctionParameters

opt_result = tuple[float, TargetFunctionParameters]


class OptimizationPolicy(ParameterizedClass):
    def __init__(self, max_num_function_evaluations: int):
        self.max_num_function_evaluations = max_num_function_evaluations
        self._params = {'max_num_function_evaluations': max_num_function_evaluations, }

    def __repr__(self):
        return self.__class__.__name__

    def _is_new_opt(self, target_function, target_value, current_target_opt):
        # return (target_function.direction == 'min' and target_value < current_target_opt) or \
        #     (target_function.direction == 'max' and target_value > current_target_opt)
        if target_function.direction == 'min':
            return target_value < current_target_opt
        elif target_function.direction == 'max':
            return target_value > current_target_opt

    def _get_init_target_opt(self, target_function: TargetFunction):
        return float('inf') if target_function.direction == 'min' else -float('inf')

    @abstractmethod
    def optimize(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                 target_function: TargetFunction) -> opt_result:
        pass


class _StaticSearch(OptimizationPolicy):
    """
    Searches the space independently of the evaluations. Subclasses are Random and GridSearch
    """

    def __init__(self, max_num_function_evaluations: int):
        super().__init__(max_num_function_evaluations)

    def optimize(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                 target_function: TargetFunction) -> opt_result:
        suggestions = self._generate_suggestions(instance, auction_request_pool, target_function)
        target_opt, target_opt_params = self._evaluate_suggestions(instance, auction_request_pool, target_function,
                                                                   suggestions)
        return target_opt, target_opt_params

    @abstractmethod
    def _generate_suggestions(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                              target_function: TargetFunction) -> list[dict[str, float]]:
        pass

    def _evaluate_suggestions(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                              target_function: TargetFunction, suggestions: list[dict[str, float]]):
        carrier_idx = target_function._carrier_model._carrier_idx
        target_opt = self._get_init_target_opt(target_function)
        target_opt_params = None
        history = []
        # parent_group_id = mlflow.active_run().data.tags['group_id']
        # with mlflow.start_run(nested=True, run_name=f'Carrier {target_function._carrier_model._carrier_idx}', ):
        # mlflow.log_params({
        #     'OptimizationPolicy': self.__class__.__name__,
        #     'CarrierModel': target_function._carrier_model._carrier_idx,
        #     'target_function_direction': target_function.direction,
        #     'target_function_error_function': target_function.error_func.__name__,
        #     'num_samples': target_function._num_samples,
        #     'max_num_function_evaluations': self.max_num_function_evaluations,
        # })

        # Iterate over the parameter combinations
        for step, suggestion in enumerate(suggestions):
            if step >= self.max_num_function_evaluations:
                break

            # Evaluate the target function
            target_value = target_function(instance, auction_request_pool, **suggestion)

            # Check if this is a new optimal value
            if self._is_new_opt(target_function, target_value, target_opt):
                target_opt = target_value
                target_opt_params = suggestion

            # log to mlflow
            mlflow.log_metric(f'target_{carrier_idx}', target_value, step=step)
            mlflow.log_metric(f'target_opt_{carrier_idx}', target_opt, step=step)

            # Register the history
            history.append({'params': suggestion, 'target': target_value})
            # log metrics
            # mlflow.log_metric('target', value=target_value, step=step)
            # mlflow.log_metric('target_opt', value=target_opt, step=step)

        mlflow.log_metric(f'target_opt_final_{carrier_idx}', target_opt)
        return target_opt, target_opt_params


class TrulyRandomSearch(_StaticSearch):
    """
    samples max_num_function_evaluations random parameter configurations
    """

    def __init__(self, max_num_function_evaluations: int):
        super().__init__(max_num_function_evaluations)
        self.rng = np.random.default_rng()
        self._params['optimization_type'] = 'global'

    def _generate_suggestions(self, instance, auction_request_pool: tuple[Request], target_function: TargetFunction) -> \
            list[dict[str, float]]:
        suggestions = []
        for _ in range(self.max_num_function_evaluations):
            suggestion = {param: self.rng.uniform(*target_function.pbounds[param]) for param in target_function.pnames}
            suggestions.append(suggestion)
        return suggestions


class GridSearch(_StaticSearch):
    """
    clearly makes not a lot of sense because params are all in the same space and there is no benefit for them to be
    on the same coordinates.
    """

    def __init__(self, max_num_function_evaluations: int):
        super().__init__(max_num_function_evaluations)
        self._params['optimization_type'] = 'global'

    def _generate_suggestions(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                              target_function: TargetFunction) -> list[dict[str, float]]:
        # Generate the grid of parameters
        num_options_per_param = math.floor(self.max_num_function_evaluations ** (1 / len(target_function.pnames)))
        # FIXME this produces weird params if num_options_per_params <= 2
        param_grid = {param: np.linspace(bounds[0], bounds[1], num=num_options_per_param)
                      for param, bounds in target_function.pbounds.items()}
        param_combinations = list(product(*param_grid.values()))
        suggestions = []
        for param_combination in param_combinations:
            suggestion = dict(zip(target_function.pnames, param_combination))
            suggestions.append(suggestion)
        return suggestions


class MyUpperConfidenceBound:
    def __init__(self, kappa: float):
        self.kappa = kappa
        self._params = {'kappa': kappa}

    def get_acquisition_function(self):
        return UpperConfidenceBound(kappa=self.kappa)

    def __repr__(self):
        return self.get_acquisition_function().__repr__()

    def __call__(self, *args, **kwargs):
        raise TypeError(f'{self.__class__.__name__} cannot be called. Get the real acquisition function using'
                        f' .get_acquisition_function() first and call that one.')


class BayesianOptimizationPolicy(OptimizationPolicy):
    def __init__(self, max_num_function_evaluations: int, init_points: int = 5):
        """

        :param init_points:
        :param max_num_function_evaluations: TOTAL number of allowed target function evaluations. During optimization,
        first, init_points random points are evaluated before the Gaussian Processes kick in. From then,
        max_num_function_evaluations - init_points are still allowed
        """
        super().__init__(max_num_function_evaluations)
        self.init_points = init_points
        # TODO: make acquisition function an argument of the constructor
        self.acquisition_function = MyUpperConfidenceBound(kappa=2.576)
        self._params['init_points'] = init_points
        self._params['acquisition_function'] = self.acquisition_function.__class__.__name__
        self._params['optimization_type'] = 'global'

    def optimize(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                 target_function: TargetFunction) -> opt_result:
        carrier_idx = target_function._carrier_model._carrier_idx
        original_direction = target_function.direction
        if original_direction == 'min':  # invert because bayes_opt always maximizes
            target_function = target_function.get_inverse()

        target_opt = self._get_init_target_opt(target_function)
        target_opt_params = None
        num_iterations = 0

        def _target_function(**params):
            nonlocal instance, auction_request_pool, target_opt, target_opt_params, carrier_idx, num_iterations
            target = target_function(instance=instance, auction_request_pool=auction_request_pool, **params)

            if self._is_new_opt(target_function, target, target_opt):
                target_opt = target
                target_opt_params = params

            # logging (with min/max inversion if necessary)
            log_target = target if original_direction == 'max' else -target
            log_target_opt = target_opt if original_direction == 'max' else -target_opt
            mlflow.log_metric(f'target_{carrier_idx}', value=log_target, step=num_iterations)
            mlflow.log_metric(f'target_opt_{carrier_idx}', log_target_opt, step=num_iterations)

            num_iterations += 1
            return target

        optimizer = BayesianOptimization(
            f=_target_function,
            pbounds=target_function.pbounds,
            acquisition_function=self.acquisition_function.get_acquisition_function(),
            random_state=None,
            verbose=0,
            allow_duplicate_points=False
        )
        optimizer.maximize(self.init_points, self.max_num_function_evaluations - self.init_points)

        """        
            if original_direction == 'min':
            # re-invert to go back to the original interpretation of the target function
            target_function = target_function.get_inverse()
            res = []
            for probe in optimizer.res:
                res.append({'params': probe['params'], 'target': - probe['target']})
        else:
            res = optimizer.res

        # logging AFTER the optimization because BO 
        for step, probe in enumerate(res):
            if self._is_new_opt(target_function, probe['target'], target_opt):
                target_opt = probe['target']
                target_opt_params = probe['params']
            mlflow.log_metric(f'target_{carrier_idx}', value=probe['target'], step=step)
            mlflow.log_metric(f'target_opt_{carrier_idx}', target_opt, step=step)
        """
        # logging (with min/max inversion if necessary)
        log_target_opt_final = target_opt if original_direction == 'max' else -target_opt
        mlflow.log_metric(f'target_opt_final_{carrier_idx}', log_target_opt_final)
        return target_opt, target_opt_params


class Nlopt(OptimizationPolicy):
    def __init__(self, max_num_function_evaluations: int, algorithm, algorithm_parameters=None,
                 lexicographic_ordering_constraint: bool = False):
        if lexicographic_ordering_constraint:
            if algorithm not in [nlopt.LN_COBYLA, nlopt.GN_AGS, nlopt.GN_ORIG_DIRECT, nlopt.GN_ISRES]:
                raise ValueError(f'Lexicographic Ordering Constraint (or any nonlinear inequality constraint for that '
                                 f'matter) only works with ISRES, AGS, ORIG_DIRECT, and COBYLA algorithms. {algorithm}'
                                 f'was provided')

        super().__init__(max_num_function_evaluations)
        if algorithm_parameters is None:
            algorithm_parameters = {}
        self._algorithm = algorithm
        self._algorithm_parameters = algorithm_parameters
        self._lexicographic_ordering_constraint = lexicographic_ordering_constraint
        self._rng = np.random.default_rng()

        # override the ParameterizedClass's 'name' parameter
        name = [name for name in dir(nlopt) if getattr(nlopt, name) == algorithm][0]
        # name += '(LOC)' if lexicographic_ordering_constraint else ''
        # name += f'({algorithm_parameters})' if algorithm_parameters else ''
        self._params['name'] = name
        self._params.update(algorithm_parameters)
        self._params['lexicographic_ordering_constraint'] = lexicographic_ordering_constraint
        self._params['optimization_type'] = 'global' if name.startswith('G') else 'local'

    def optimize(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                 target_function: TargetFunction) -> opt_result:
        carrier_idx = target_function._carrier_model._carrier_idx
        optimizer = nlopt.opt(self._algorithm, len(target_function.pnames))
        target_opt = self._get_init_target_opt(target_function)
        target_opt_params = None
        # For now, I need to track the number of target function calls manually because of a bug in PRAXIS, see:
        #  https://github.com/stevengj/nlopt/issues/606
        num_iterations = 0

        def _target_function(x, grad):
            """
            My TargetFunction class accepts only keyword arguments, but nlopt works with arrays. Thus, I need to
            use this wrapper function to convert the array to keyword arguments.
            This also gives me the opportunity to log the target function values and the target_opt value.
            """
            nonlocal carrier_idx, target_opt, target_opt_params, num_iterations
            if grad.size > 0:
                raise ValueError('only use this with derivative-free optimizers')
            params = TargetFunctionParameters.from_numpy(x)
            target = target_function(instance=instance, auction_request_pool=auction_request_pool, **params)
            if self._is_new_opt(target_function, target, target_opt):
                if not self._lexicographic_ordering_constraint or _lexicographic_ordering(x, grad) <= 0:
                    # either no LOC is in place OR the LOC is satisfied  --> update the target_opt
                    target_opt = target
                    target_opt_params = params

            mlflow.log_metric(f'target_{carrier_idx}', target, step=num_iterations)
            mlflow.log_metric(f'target_opt_{carrier_idx}', target_opt, step=num_iterations)
            num_iterations += 1
            return target

        if target_function.direction == 'max':
            optimizer.set_max_objective(_target_function)
        elif target_function.direction == 'min':
            optimizer.set_min_objective(_target_function)
        optimizer.set_lower_bounds([target_function.pbounds[param][0] for param in target_function.pnames])
        optimizer.set_upper_bounds([target_function.pbounds[param][1] for param in target_function.pnames])

        def _lexicographic_ordering(x, grad):
            """
            Constraint: Ensures that coordinates are ordered lexicographically.
            NLopt expects constraints in the form g(x) <= 0. Thus, a positive value indicates a violation.
            """
            if grad.size > 0:
                raise ValueError('only use this with derivative-free optimizers')
            # compute the constraint violations
            sum_constraint_violation = 0
            num_coords = len(x) // 2
            for i in range(num_coords - 1):
                x_diff = x[i] - x[i + 1]
                if x_diff > 0:
                    sum_constraint_violation += x_diff
                elif x_diff == 0:
                    y_diff = x[i + num_coords] - x[i + num_coords + 1]
                    if y_diff > 0:
                        sum_constraint_violation += y_diff
            return sum_constraint_violation

        if self._lexicographic_ordering_constraint:
            optimizer.add_inequality_constraint(_lexicographic_ordering, 1e-8)  # tol parameter is not a kwarg

        optimizer.set_maxeval(self.max_num_function_evaluations)
        # optimizer.set_ftol_abs(1e-8)  # do not use, some algorithm will terminate before maxeval is reached otherwise
        # optimizer.set_initial_step()

        x0 = np.array([self._rng.uniform(*target_function.pbounds[param]) for param in target_function.pnames])

        try:
            target_opt_params: np.array = optimizer.optimize(x0)
            target_opt_params = TargetFunctionParameters.from_numpy(target_opt_params)
        except nlopt.RoundoffLimited as e:
            # https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#error-codes-negative-return-values:
            # -4: Halted because roundoff errors limited progress. (In this case, the optimization still typically
            # returns a useful result.)
            print(f'NLopt: RoundoffLimited Error: {e} after {optimizer.get_numevals()} evaluations. Continuing '
                  f'with the current best solution: x={target_opt_params}, target={target_opt}')
            target_opt = optimizer.last_optimum_value()

        if target_opt != optimizer.last_optimum_value():
            # Happens only with nlopt.GN_AGS and if the very last evaluation is an improvement. Then, NLopt does
            # not properly recognize that the target_opt has changed. https://github.com/stevengj/nlopt/issues/603
            # warnings.warn(
            #     f"NLopt: manually tracked target_opt is not the same as NLopt's last_optimum_value. "
            #     f"Algorithm: {optimizer.get_algorithm_name()}, parameters: {self._algorithm_parameters}, LOC: {self._lexicographic_ordering_constraint}."
            #     f"\ntarget_opt (manual)={target_opt}"
            #     f"\nlast_optimum_value (NLopt)={optimizer.last_optimum_value()}\n"
            #     f"Continnuing with NLopt's value."
            # )
            # NOTE disregard my own target_opt tracking, because nlopt does it itself!
            target_opt = optimizer.last_optimum_value()

        return_code = optimizer.last_optimize_result()
        if return_code < 0:
            warnings.warn(f'NLopt: Optimizer did not return successfully. return_code={return_code}!')

        if return_code != nlopt.MAXEVAL_REACHED:
            warnings.warn(f'NLopt: Optimizer did not reach the maximum number of evaluations. '
                          f'return_code={return_code}!')

        mlflow.log_metric(f'target_opt_final_{carrier_idx}', target_opt)
        return target_opt, target_opt_params
