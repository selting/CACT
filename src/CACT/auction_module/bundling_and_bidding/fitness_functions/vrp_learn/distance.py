import numpy as np
from pyvrp import ProblemData, Client, Depot, VehicleType, Model
from pyvrp.stop import NoImprovement
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from shapely.geometry import MultiPoint, Polygon


def _my_directed_hausdorff_distance(A, B, norm=2):
    """
    The directed Hausdorff distance H between two point sets A and B is the maximum of distances between each point x in A to its nearest neighbor y in B.
    :param A: 2D set of points
    :param B: 2D set of points
    :return:
    """
    pairwise_dist = cdist(A, B, metric='minkowski', p=norm)
    nearest_neighbor_dist = np.min(pairwise_dist, axis=1)
    max_ = np.max(nearest_neighbor_dist)
    return max_


def my_hausdorff_distance(A, B, norm=2):
    """

    :param A: 2D set of points
    :param B: 2D set of points
    :return:
    """
    return max(_my_directed_hausdorff_distance(A, B, norm), _my_directed_hausdorff_distance(B, A, norm))


def _my_d_six(A, B, norm=2):
    """
    d_6 distance from this paper: Dubuisson, M.-P., and A.K. Jain. “A Modified Hausdorff Distance for Object Matching.”
    In Proceedings of 12th International Conference on Pattern Recognition, 1:566–68 vol.1, 1994.
    https://doi.org/10.1109/ICPR.1994.576361.

    :param A:
    :param B:
    :param norm:
    :return:
    """
    pairwise_dist = cdist(A, B, metric='minkowski', p=norm)
    nearest_neighbor_dist = np.min(pairwise_dist, axis=1)
    d_six = 1 / len(A) * nearest_neighbor_dist.sum()
    return d_six


def my_modified_hausdorff_distance(A, B, norm=2):
    """
    modified hausdorff distance as defined in: Dubuisson, M.-P., and A.K. Jain. “A Modified Hausdorff Distance for
    Object Matching.” In Proceedings of 12th International Conference on Pattern Recognition, 1:566–68 vol.1, 1994.
    https://doi.org/10.1109/ICPR.1994.576361.

    :param A:
    :param B:
    :param norm:
    :return:
    """
    return max(_my_d_six(A, B, norm), _my_d_six(B, A, norm))


def my_convex_hull_jaccard_distance(A, B):
    """
    1 - jaccard index. Jaccard index is the intersection over union of the two convex hulls of A and B

    :param A: set of points
    :param B: set of points
    :return:
    """
    A_hull = MultiPoint(A).convex_hull
    B_hull = MultiPoint(B).convex_hull
    intersection = A_hull.intersection(B_hull)
    union = A_hull.union(B_hull)
    jaccard_index = intersection.area / union.area
    jaccard_distance = 1 - jaccard_index
    # if jaccard_index == 0:
    #     pass
    # plt.plot(*A.T, 'o')
    # plt.plot(*B.T, 'o')
    # # plt.plot(*np.array(intersection.exterior.coords).T, '-o')  # intersection empty
    # plt.show()
    return jaccard_distance


def my_MinWBMP(A, B, norm=2):
    """
    Solves the minimum weighted bipartite matching problem to match points in A (workers) to points in B (jobs), also
    known as the assignment problem.
    The cost of assigning a to b is the distance between the points


    :param A: 2D set of points
    :param B: 2D set of points
    :param norm: minkowski distance norm
    :return:
    """
    cost = cdist(A, B, metric='minkowski', p=norm)
    row_idx, col_idx = linear_sum_assignment(cost)
    obj_val = cost[row_idx, col_idx].sum()
    return obj_val


def _solve_tsp(vertices: np.ndarray, cost: np.ndarray, scale_factor: float = 1e4):
    vertices = (vertices * scale_factor).astype(int)
    cost = (cost * scale_factor).astype(int)

    clients = [Client(*v) for v in vertices[1:]]
    depot = Depot(*vertices[0])
    vehicle = VehicleType(unit_distance_cost=1, unit_duration_cost=1)
    pdata = ProblemData(clients, [depot], [vehicle], distance_matrices=[cost], duration_matrices=[cost])
    model = Model.from_data(pdata)
    result = model.solve(stop=NoImprovement(100), display=False)
    return result.best


def my_tsp_obj_val_diff(A, B, norm=2):
    """
    Solves the TSP for A and B, and returns the absolute difference in their objective function value.
    Currently, uses the PyVrp solver

    :param A:
    :param B:
    :param norm:
    :return:
    """
    A_cost = squareform(pdist(A, metric='minkowski', p=norm))
    B_cost = squareform(pdist(B, metric='minkowski', p=norm))
    # ideally, this should be solved using the already implemented static routing solvers...
    scale_factor = 1e4

    A_sol = _solve_tsp(A, A_cost, scale_factor)
    B_sol = _solve_tsp(B, B_cost, scale_factor)
    A_obj_val = A_sol.distance()
    B_obj_val = B_sol.distance()
    dist = abs(A_obj_val - B_obj_val) / scale_factor
    return dist


def my_tsp_hull_jaccard_distance(A, B, norm=2):
    """
    instead of taking the convex hull, create polygons from the tsp solutions and compute their jaccard index
    :return:
    """
    A_cost = squareform(pdist(A, metric='minkowski', p=norm))
    B_cost = squareform(pdist(B, metric='minkowski', p=norm))
    # ideally, this should be solved using the already implemented static routing solvers...
    scale_factor = 1e4

    A_sol = _solve_tsp(A, A_cost, scale_factor)
    B_sol = _solve_tsp(B, B_cost, scale_factor)

    A_tour = np.concatenate([A[[0]], A[A_sol.routes()[0].visits()], A[[0]]])
    A_poly = Polygon(A_tour)
    B_tour = np.concatenate([B[[0]], B[B_sol.routes()[0].visits()], B[[0]]])
    B_poly = Polygon(B_tour)
    # plt.plot(*A_poly.exterior.xy, '-o')
    # plt.plot(*B_poly.exterior.xy, '-o')
    # plt.show()
    union = A_poly.union(B_poly)
    intersection = A_poly.intersection(B_poly)
    jaccard_dist = 1 - (intersection.area / union.area)
    return jaccard_dist


def my_dice_similarity_coefficient(A, B, grid_resolution: float = 1e-2):
    """
    The DSC is a measure for similarity between two samples. Here, we implement it as detailed in:
    https://doi.org/10.1109/TPAMI.2015.2408351.

    "consider an imaginary grid on the union of the two point sets and calculate the overlap (intersection) between
    the point sets with respect to the grid. Points are assigned to subsets depending on whether they are or are not
    in the intersection to build the confusion matrix."

    :param A:
    :param B:
    :param grid_resolution:
    :return:
    """
    # generate the grid
    min_ = np.concatenate((A, B), axis=0).min(axis=0)
    max_ = np.concatenate((A, B), axis=0).max(axis=0)
    x = np.arange(min_[0], max_[0], grid_resolution)
    y = np.arange(min_[1], max_[1], grid_resolution)
    grid = np.meshgrid(x, y)
    # check
    pass
    # ...


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    A = rng.uniform(0, 25, (4, 2))
    B = rng.uniform(0, 25, (4, 2))
    my_tsp_hull_jaccard_distance(A, B)
