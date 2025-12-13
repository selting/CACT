from copy import copy

import numpy as np

from core_module.instance import CAHDInstance
from core_module.request import Request
from utility_module.parameterized_class import ParameterizedClass


class TargetFunctionParameters(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_numpy(self):
        """
        Convert the x, y parameter dict to a 2D numpy array
        :return:
        """
        return np.array([[self[f'x{i}'], self[f'y{i}']] for i in range(len(self) // 2)])

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        """
        Convert a numpy array to an x, y parameter dict.
        Accepts either a 2D array with shape (n, 2) or a 1D array with shape (2n,). In the latter case, the array is
        reshaped to (n, 2).

        :param arr:
        :return:
        """
        if arr.ndim == 1:
            arr = arr.reshape(2, -1).T

        return cls({f'x{i}': arr[i, 0] for i in range(arr.shape[0])} |
                   {f'y{i}': arr[i, 1] for i in range(arr.shape[0])})


class TargetFunction(ParameterizedClass):
    def __init__(self,
                 error_func: callable,
                 carrier_model,
                 X,
                 y_true,
                 direction: str,
                 target_func_pnames: tuple[str],
                 target_func_pbounds):
        """

        :param error_func: must accept two arguments y and y_pred
        :param target_func_pnames: the names of the model parameters that the target function will optimize
        :param target_func_pbounds: bounds of the target_func parameters, either a string ('vienna', 'euclidean') or a
        mapping of string parameter names to their (min, max) bounds
        :param direction: either 'min' or 'max'
        """
        self.error_func = error_func
        self._carrier_model = carrier_model
        self._X = X
        self._y_true = y_true
        self._num_samples = len(X)
        self.direction = direction
        self.pnames = target_func_pnames
        if 'vienna' in target_func_pbounds:
            self.pbounds = {f'x{i}': (48.116600, 48.323843) for i in range(len(target_func_pnames) // 2)} | \
                           {f'y{i}': (16.174965, 16.579399) for i in range(len(target_func_pnames) // 2)}
        elif target_func_pbounds == 'euclidean':
            self.pbounds = {f'x{i}': (0, 25) for i in range(len(target_func_pnames) // 2)} | \
                           {f'y{i}': (0, 25) for i in range(len(target_func_pnames) // 2)}
        else:
            self.pbounds = target_func_pbounds

        self._params = {
            'error_func': str(self.error_func),
            'carrier_model': str(self._carrier_model),
            'num_samples': str(self._num_samples),
            'direction': str(self.direction),
        }

    def __repr__(self):
        return f'{self.error_func.__name__}'

    def __call__(self, instance: CAHDInstance, auction_request_pool: tuple[Request], **model_params):
        if not model_params:
            raise ValueError(f'no model_params provided. model_params={self.pnames}')
        if not all(k in self.pnames for k in model_params):
            raise ValueError('model_params must contain all model parameters')
        # NOTE: previously i was checking bounds, but many methods also search outside the bounds. The final result
        #  will always be feasible, i.e. inside the bounds.

        # for param, value in model_params.items():
        #     if value < self.pbounds[param][0] or value > self.pbounds[param][1]:
        #         raise ValueError(
        #             f'model parameter {param} is out of bounds. value={value}, bounds={self.pbounds[param]}')
        current_params_cache = copy(self._carrier_model.current_params)
        self._carrier_model.current_params = model_params
        y_pred = self._carrier_model.compute_bid_on_bundles(instance, self._X)
        self._carrier_model.current_params = current_params_cache
        # TODO the .compute_bid_on_bundles may return 'infeasible' for some bids. This cannot be evaluated yet.
        target_value = self.error_func(self._y_true, y_pred)
        return target_value

    def get_inverse(self):
        """
        Get the inverse of this target function, i.e. the target function with the opposite direction and the negative
        error function.
        :return:
        """

        def neg_error_func(y_true, y_pred):
            return -self.error_func(y_true, y_pred)

        inv_direction = 'min' if self.direction == 'max' else 'max'
        return TargetFunction(neg_error_func, self._carrier_model, self._X, self._y_true, inv_direction, self.pnames,
                              self.pbounds)
