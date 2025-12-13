from collections.abc import MutableMapping
from abc import ABC
import datetime as dt
from numbers import Number
from types import FunctionType
from abc import ABCMeta


def my_value_parser(obj, parse_datetime=False):
    """
    converts an object to a loggable format (mostly for mlflow)
    numbers stay numbers, datetime are converted to strings, functions to their names, classes to their names,
    bools stay bools, strings stay strings, everything else is converted to a string.

    """
    if parse_datetime:
        if isinstance(obj, dt.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, dt.timedelta):
            return obj.total_seconds()
    # checking for all sorts of functions, but excluding classes that implement __call__()
    if isinstance(obj, FunctionType):
        return obj.__name__
    if callable(obj):
        if not hasattr(obj, '__call__'):
            return obj.__name__
    elif isinstance(obj, ABCMeta):
        return obj.__name__
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, Number):
        return obj
    elif isinstance(obj, list):
        return '[' + ', '.join([my_value_parser(o) for o in obj]) + ']'
    # elif isinstance(obj, dict):
    #     return {k: my_logify(v) for k, v in obj.items()}
    elif obj is None:
        return obj
    return str(obj)


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


class ParameterizedClass(ABC):
    """
    A base class for classes that have a ._params attribute and need
    a .params property to retrieve their parameters as a nested dictionary.
    """

    @property
    def params(self):
        """
        Returns the parameters of the class as a nested dictionary.
        """
        if ParameterizedClass not in self.__class__.__bases__:
            params = {'name': self.__class__.__name__}
        else:
            params = {}
        params.update(self._get_params(self))
        return params

    def _get_params(self, instance):
        """
        Recursively retrieves the parameters of the instance and all its attributes that are instances of
        ParameterizedClass. Returns a nested dictionary. Values are parsed with my_value_parser.
        """
        params_dict = {}
        if hasattr(instance, '_params') and isinstance(instance._params, dict):
            for key, value in instance._params.items():
                if isinstance(value, ParameterizedClass):
                    params_dict[key] = value.params
                else:
                    params_dict[key] = my_value_parser(value, True)
        return params_dict
