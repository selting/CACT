import random
from abc import abstractmethod
from typing import Union, Sequence
from warnings import simplefilter

import numpy as np
import torch
from utility_module.parameterized_class import ParameterizedClass
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.partition_based.partition import Partition
from auction_module.bundling_and_bidding.type_defs import QueriesType, ResponsesType
from core_module.instance import CAHDInstance
from core_module.request import Request

# supress the ConvergenceWarning from sklearn on the HPC
# if socket.gethostname() == 'hpc3':
simplefilter("ignore", category=ConvergenceWarning)

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FitnessFunction(ParameterizedClass):
    def __init__(self):
        self.search_space = None
        self._params = {}
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @abstractmethod
    #  TODO let elements be given in their class type not as binaries
    def __call__(self, instance: CAHDInstance, elements: Union[Sequence[Bundle], Sequence[Partition], Sequence[Assignment]], **kwargs):
        pass

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType) -> dict | None:
        pass

    @property
    def fittable(self):
        return "fit" in self.__class__.__dict__


class FitnessRandom(FitnessFunction):
    def __repr__(self):
        return f'{self.__class__.__name__}'

    def __call__(self, instance: CAHDInstance, elements: Union[Sequence[Bundle], Sequence[Partition], Sequence[Assignment]], **kwargs):
        return [random.random() for _ in range(self.instance.num_carriers)]


def MinWBMP(x):
    """
    Return the objective of the Minimum Weighted Bipartite Matching of the matrix x.
    If x contains NaN values, they are replaced with infinity to not be considered in the matching unless all values
    are NaN.

    :param x:
    :return:
    """
    x_adj = np.copy(x).astype(float)
    x_adj[np.isnan(x_adj)] = np.infty
    try:
        row_idx, col_idx = linear_sum_assignment(x_adj)
        objVal = x_adj[row_idx, col_idx].sum()
    except ValueError as e:
        objVal = np.infty
    return objVal


def nested_train_test_split(X, y, train_size=None, test_size=None, random_state=None):
    """
    Splits nested data into training and testing subsets.

    This function assumes that the input data `X` is a list of lists (or similar nested structure),
    and that `y` is a list of the same outer length as `X`. The function splits each inner list in `X`
    and the corresponding element in `y` into train and test subsets, maintaining the pairing between `X` and `y`.

    Parameters:
    X (list of lists): The input features to split. Each inner list is considered as a separate dataset.
    y (list): The target variables to split. Each element corresponds to an inner list in `X`.
    test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
                               to include in the test split. If int, represents the absolute number of test samples.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split.
                                  Pass an int for reproducible output across multiple function calls.

    Returns:
    X_train (list of lists): The training subsets of `X`.
    X_test (list of lists): The testing subsets of `X`.
    y_train (list): The training subsets of `y`.
    y_test (list): The testing subsets of `y`.
    """
    # if only one sample is given, return it as training sample and empty lists for test samples
    if len(X[0]) == 1:
        X_train = X
        X_test = [[] for _ in range(len(X))]
        y_train = y
        y_test = [[] for _ in range(len(X))]

    else:
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for X_inner, y_inner in zip(X, y):
            X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(
                X_inner, y_inner, train_size=train_size, test_size=test_size, random_state=random_state,
            )
            X_train.append(X_inner_train)
            X_test.append(X_inner_test)
            y_train.append(y_inner_train)
            y_test.append(y_inner_test)

    return X_train, X_test, y_train, y_test
