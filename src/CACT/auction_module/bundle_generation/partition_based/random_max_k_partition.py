# https://stackoverflow.com/a/45880095/15467861 this solutions finds a random *k-partition* of a set. I.e. the user
#  specifies the number of subsets to partition the set into.
#  What i want is to sample uniformly from the set of all partitions of a set of AT MOST k subsets. Knowing the number
#  of ways to partition a set into k subsets, we can use this solution to first sample k according to that number (
#  Stirling number of the second kind) and then sample uniformly from the set of all k-partitions of the set.


import random

import numpy as np

from utility_module.combinatorics import nCr, stirling_second


def ith_subset(A, k, i):
    """
    (Return ith k-subset of A)
    Returns the first subset from the ith k-subset partition of A

    """
    # Choose first element x
    n = len(A)
    if n == k:
        return A
    if k == 0:
        return []
    for x in range(n):
        # Find how many cases are possible with the first element being x
        # There will be n-x-1 left over, from which we choose k-1
        extra = nCr(n - x - 1, k - 1)
        if i < extra:
            break
        i -= extra
    return [A[x]] + ith_subset(A[x + 1 :], k - 1, i)


def gen_part(A, k, i):
    """Return i^th k-partition of elements in A (zero-indexed) as list of lists"""
    if k == 1:
        return [A]
    n = len(A)
    # First find appropriate value for y - the extra amount in this subset
    for y in range(0, n - k + 1):
        extra = stirling_second(n - 1 - y, k - 1) * nCr(n - 1, y)
        if i < extra:
            break
        i -= extra
    # We count through the subsets, and for each subset we count through the partitions
    # Split i into a count for subsets and a count for the remaining partitions
    count_partition, count_subset = divmod(i, nCr(n - 1, y))
    # Now find the i^th appropriate subset
    subset = [A[0]] + ith_subset(A[1:], y, count_subset)
    S = set(subset)
    return [subset] + gen_part([a for a in A if a not in S], k - 1, count_partition)


def random_k_partition(A, k):
    """
    generate a single random partition of a list A into k non-empty subsets
    :param A:
    :param k:
    :return:
    """
    stirling2 = stirling_second(len(A), k)
    i = random.randrange(0, stirling2)
    return gen_part(A, k, i)


def random_max_k_partition(A, max_k: int):
    """
    generate a single random partition of a list A into at most k non-empty subsets.
    First, a k is chosen randomly between 1 and max_k, the probability of each k is proportional to the number of
    partitions of A into k subsets, i.e. the Stirling number of the second kind.
    Then, a random partition of A into k subsets is generated.
    :param A:
    :param max_k:
    :return:
    """
    k_stirling = {k_: stirling_second(len(A), k_) for k_ in range(1, max_k + 1)}
    k_probabilities = {k: v / sum(k_stirling.values()) for k, v in k_stirling.items()}
    k_choices, k_probabilities = zip(*k_probabilities.items())
    k = np.random.choice(k_choices, replace=False, p=k_probabilities)

    return random_k_partition(list(A), k)


if __name__ == "__main__":
    A = list(range(30))
    max_k = 3
    print(random_max_k_partition(A, max_k))

    # TEST WHETHER DISTRIBUTION IS AS EXPECTED ================================================================
    # generated_sizes = dict()
    # for _ in trange(100_000):
    #     part = random_max_k_partition(A, 3)
    #     size = len(part)
    #     if size not in generated_sizes:
    #         generated_sizes[size] = 0
    #     generated_sizes[size] += 1
    # pprint(generated_sizes)
    # pprint({k: v / sum(generated_sizes.values()) for k, v in generated_sizes.items()})
    # print('---')
    #
    # from sympy.functions.combinatorial.numbers import stirling
    # stirling_numbers = {k: stirling(len(A), k, kind=2) for k in range(1, max_k + 1)}
    # pprint(stirling_numbers)
    # pprint({k: float(v / sum(stirling_numbers.values())) for k, v in stirling_numbers.items()})
