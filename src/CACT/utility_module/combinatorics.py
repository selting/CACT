import itertools

from sympy.functions.combinatorial.numbers import stirling


def algorithm_u(ns, m):
    """
    https://codereview.stackexchange.com/a/1944

    Generates all set partitions with a given number of blocks. The total amount of k-partitions is given by the
    Stirling number of the second kind

    :param ns: sequence of integers to build the subsets from
    :param m: integer, smaller than ns, number of subsets
    :return:
    """
    assert m > 1

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


def size_k_partitions(collection, k):
    """
    Generate all partitions of the collection with k subsets. The total amount of k-partitions is given by the Stirling
    number of the second kind.

    https://codereview.stackexchange.com/a/240277/252285

    Note This is faster than algorithm_u

    Example:
    list(size_k_partitions([1, 2, 3, 4], 2)
    [
    [[1], [2, 3, 4]],
    [[1, 2], [3, 4]],
    [[2], [1, 3, 4]],
    [[1, 2, 3], [4]],
    [[2, 3], [1, 4]],
    [[1, 3], [2, 4]],
    [[3], [1, 2, 4]]
    ]


    :param collection:
    :param k:
    :return:
    """
    yield from _size_k_partitions(collection, k, k)


def _size_k_partitions(collection, min_, k):
    """
    auxiliary function for size_k_partitions

    :param collection:
    :param min_:
    :param k:
    :return:
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in _size_k_partitions(collection[1:], min_ - 1, k):
        if len(smaller) > k:
            continue
        # insert `first` in each of the subpartition's subsets
        if len(smaller) >= min_:
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        if len(smaller) < k:
            yield [[first]] + smaller


def max_size_k_partitions(collection, k):
    """
    generates all partitions that have a size of 1, 2, ..., k bundles

    :param collection:
    :param k:
    :return:
    """
    partitions = []
    for k_ in range(1, k + 1):
        k_partitions = size_k_partitions(collection, k_)
        partitions.extend(k_partitions)
    return partitions


def power_set(s, include_empty_set=True):
    """
    set of all subsets of s, including the empty set and s itself

    Example:
        power_set([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(s)
    if include_empty_set:
        range_ = range(len(s) + 1)
    else:
        range_ = range(1, len(s) + 1)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range_)


# def random_max_k_partition_idx(ls, max_k) -> list[int]:
#     """partition ls into at most k randomly sized disjoint subsets"""
#     # if random.getstate()[1][0] == 0:  # Check if the random number generator has been seeded
#     #     random.seed(1680)
#     # https://stackoverflow.com/a/45880095
#     raise DeprecationWarning('this function is not correct. check Bundle().random_max_k() class method')
#     if max_k < 1:
#         return []
#     # randomly determine the actual k
#     k = random.randint(1, min(max_k, len(ls)))  # FIXME does not consider different probabilities for different k?
#     # We require that this list contains k different values, so we start by adding each possible different value.
#     indices = list(range(k))
#     # now we add random values from range(k) to indices to fill it up to the length of ls
#     indices.extend([random.choice(list(range(k))) for _ in range(len(ls) - k)])
#     # shuffle the indices into a random order
#     random.shuffle(indices)
#     return indices


def average_num_blocks(n, max_k):
    """
    A set of n elements can be partitioned into k blocks in S(n,k) ways, where S(n,k) is the Stirling number of the
    second kind. This function returns the EXPECTED/AVERAGE NUMBER OF BLOCKS in a random partition of n elements into
    AT MOST k blocks.

    :param n:
    :param max_k:
    :return:
    """
    stirling_numbers = dict()
    for k in range(0, max_k + 1):  # k is the number of blocks
        stir = stirling(
            n, k, kind=2
        )  # stir is the number of partitions of n elements into k blocks
        stirling_numbers[k] = stir
    avg = sum([k * v for k, v in stirling_numbers.items()]) / sum(
        stirling_numbers.values()
    )
    return avg


def expected_number_of_distinct_outcomes(n, m):
    """
    Returns the expected number of distinct outcomes for an m- sided die rolled n times.

    Alternative interpretation: Expected number of included (i.e. non-empty) persons when assigning n items to m persons
     randomly.

    :param n: items to assign
    :param m: persons to assign to
    :return:
    """
    # both versions seem correct
    # return m * (1 - (1 - 1 / m) ** n)  # copilot
    return m * (
        1 - ((m - 1) / m) ** n
    )  # https://math.stackexchange.com/a/3974586/998526


fact = [1]


def nCr(n, k):
    """Return number of ways of choosing k elements from n"""
    while len(fact) <= n:
        fact.append(fact[-1] * len(fact))
    return fact[n] // (fact[k] * fact[n - k])


cache = {}


def stirling_second(n, k):
    """
    Return the Stirling number of the second kind, i.e. the number of ways of partitioning n items into k non-empty
    subsets
    """
    if k == 1:
        return 1
    key = n, k
    if key in cache:
        return cache[key]
    # The first element goes into the next partition
    # We can have up to y additional elements from the n-1 remaining
    # There will be n-1-y left over to partition into k-1 non-empty subsets
    # so n-1-y>=k-1
    # y<=n-k
    t = 0
    for y in range(0, n - k + 1):
        t += stirling_second(n - 1 - y, k - 1) * nCr(n - 1, y)
    cache[key] = t
    return t

