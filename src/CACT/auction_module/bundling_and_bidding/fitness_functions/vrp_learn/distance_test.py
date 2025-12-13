import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform, cdist
from shapely.geometry import MultiPoint, Polygon
from tqdm import tqdm

from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.distance import (
    my_hausdorff_distance, my_modified_hausdorff_distance, my_MinWBMP, my_convex_hull_jaccard_distance, \
    my_tsp_obj_val_diff, _solve_tsp, my_tsp_hull_jaccard_distance, _my_d_six)

plt.style.use('ggplot')


def my_plot(true, pred, depot):
    plt.plot(depot[0], depot[1], 'ks')
    plt.text(depot[0], depot[1], 'Depot')
    plt.plot(true[:, 0], true[:, 1], 'o', label='True')
    for idx, c in enumerate(true):
        plt.text(c[0], c[1], idx, color='red')
    plt.plot(pred[:, 0], pred[:, 1], 'o', label='Predicted')
    for idx, c in enumerate(pred):
        plt.text(c[0], c[1], idx, color='blue')
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def exp_random_pred(num_samples=10, distances=[my_hausdorff_distance, ]):
    """
    generate random true and predicted points for a number of samples, then compute the metrics between the pairs
    :param num_samples:
    :param distances:
    :return:
    """
    n_true = 4
    n_pred = 4
    clients_true = rng.uniform(low=(0, 0), high=(1, 1), size=(n_true, 2))
    records = []
    for sample in range(num_samples):
        record = {'sample': sample}
        clients_pred = rng.uniform(low=(0, 0), high=(1, 1), size=(n_pred, 2))
        for metric in distances:
            val = metric(clients_true, clients_pred)
            record[metric.__name__] = val
        records.append(record)
    return pd.DataFrame(records).set_index('sample', inplace=False)


def exp_increasing_add_noise(noise_levels, num_samples=10, distances=[my_hausdorff_distance, ]):
    """
    copy clients_true, add noise, then compute the metrics
    :param noise_levels:
    :param num_samples:
    :param distances:
    :return:
    """
    n_true = 4
    clients_true = rng.uniform(low=(0, 0), high=(1, 1), size=(n_true, 2))
    records = []
    for scale in tqdm(noise_levels):
        for sample in range(num_samples):
            record = {'noise_scale': scale, 'sample': sample}
            clients_pred = clients_true + rng.normal(0, scale, (n_true, 2))
            for metric in distances:
                val = metric(clients_true, clients_pred)
                record[metric.__name__] = val
            records.append(record)
    df2 = pd.DataFrame(records)
    # mdf2 = df2.melt(id_vars=['noise_scale', 'sample'], var_name='metric', value_name='value')
    # fig = px.strip(mdf2, x='noise_scale', y='value', color='metric')  # try px.box or px.violin
    gdf2 = df2.groupby(['noise_scale']).mean().drop(columns=['sample'])
    fig = px.line(gdf2, markers=True,
                  title=f'Increasing additive noise scale \n {num_samples} samples per noise_scale setting')

    fig.show()
    pass


def exp_outlier_sensitivity(distances=[my_hausdorff_distance, ]):
    """
    copy clients_true, shift only a single outlier further and further away
    """
    n_true = 4
    clients_true = rng.uniform(low=(0, 0), high=(1, 1), size=(n_true, 2))
    records = []
    experiment_data = []
    for outlier_scale in np.linspace(1, 5, 20):
        record = {'outlier_scale': outlier_scale}
        clients_pred = clients_true.copy()
        clients_pred[-1] *= outlier_scale
        experiment_data.append(clients_pred[-1])
        for dist_func in distances:
            val = dist_func(clients_true, clients_pred)
            record[dist_func.__name__] = val
        records.append(record)

    # visualizing the experiment
    clients_df = pd.DataFrame(clients_true, columns=['x', 'y'])
    fig = px.scatter(clients_df, x='x', y='y')
    exp_array = np.array(experiment_data)
    fig.add_scatter(x=exp_array[:, 0], y=exp_array[:, 1], mode='markers+lines')
    fig.show()

    # visualizing the results
    df3 = pd.DataFrame(records)
    mdf3 = df3.melt(id_vars=['outlier_scale'], var_name='metric', value_name='value')
    fig = px.line(mdf3, x='outlier_scale', y='value', color='metric')
    fig.show()
    pass


def plot_cases(cases: list, distance_funcs: list):
    num_cases = len(cases)
    fig_scale = 2.5
    fig, axs = plt.subplots(nrows=len(distance_funcs) + 1, ncols=num_cases,
                            figsize=(num_cases * fig_scale, (len(distance_funcs) + 1) * fig_scale),
                            sharex=True, sharey=True, squeeze=False,
                            subplot_kw={'aspect': 'equal'})
    for col_idx, (A, B) in enumerate(cases):
        axs[0, col_idx].plot(*A.T, '.')
        axs[0, col_idx].plot(*B.T, '.')
        for row_idx, dist_func in enumerate(distance_funcs, start=1):
            func_val = dist_func(A, B)
            ax: plt.Axes = axs[row_idx, col_idx]
            ax.text(x=1, y=1, s=f'{func_val:.3f}', transform=ax.transAxes, ha='right', va='top')
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 1)
            if dist_func.__name__ == 'my_convex_hull_jaccard_distance':
                A_hull = MultiPoint(A).convex_hull
                B_hull = MultiPoint(B).convex_hull
                intersection = A_hull.intersection(B_hull)

                ax.plot(*A.T, '.')
                ax.plot(*B.T, '.')
                try:
                    ax.fill(*A_hull.exterior.xy, alpha=0.3)
                    ax.fill(*B_hull.exterior.xy, alpha=0.3)
                    ax.fill(*intersection.exterior.xy, alpha=0.5, color='green')
                except AttributeError:
                    pass
            elif dist_func.__name__ == 'my_tsp_obj_val_diff':
                A_cost = squareform(pdist(A, metric='euclidean'))
                B_cost = squareform(pdist(B, metric='euclidean'))
                # ideally, this should be solved using the already implemented static routing solvers...
                scale_factor = 1e4
                A_sol = _solve_tsp(A, A_cost, scale_factor)
                B_sol = _solve_tsp(B, B_cost, scale_factor)
                A_tour = np.concatenate([A[[0]], A[A_sol.routes()[0].visits()], A[[0]]])
                A_poly = Polygon(A_tour)
                B_tour = np.concatenate([B[[0]], B[B_sol.routes()[0].visits()], B[[0]]])
                B_poly = Polygon(B_tour)
                try:
                    ax.plot(*A_poly.exterior.xy, '-o')
                    ax.plot(*B_poly.exterior.xy, '-o')
                except AttributeError:
                    pass
                ax.text(0, 0, s=f'{A_sol.distance() / scale_factor:.3f}', va='bottom', ha='left', color='red',
                        transform=ax.transAxes, )
                ax.text(1, 0, s=f'{B_sol.distance() / scale_factor:.3f}', va='bottom', ha='right', color='blue',
                        transform=ax.transAxes, )
            elif dist_func.__name__ == 'my_tsp_hull_jaccard_distance':
                A_cost = squareform(pdist(A, metric='euclidean'))
                B_cost = squareform(pdist(B, metric='euclidean'))
                # ideally, this should be solved using the already implemented static routing solvers...
                scale_factor = 1e4
                A_sol = _solve_tsp(A, A_cost, scale_factor)
                B_sol = _solve_tsp(B, B_cost, scale_factor)
                A_tour = np.concatenate([A[[0]], A[A_sol.routes()[0].visits()], A[[0]]])
                A_poly = Polygon(A_tour)
                B_tour = np.concatenate([B[[0]], B[B_sol.routes()[0].visits()], B[[0]]])
                B_poly = Polygon(B_tour)
                try:
                    ax.plot(*A_poly.exterior.xy, '.')
                    ax.plot(*B_poly.exterior.xy, '.')
                    ax.fill(*A_poly.exterior.xy, alpha=0.3)
                    ax.fill(*B_poly.exterior.xy, alpha=0.3)
                except AttributeError:
                    pass
                if hasattr(A_poly.intersection(B_poly), 'exterior'):
                    ax.fill(*A_poly.intersection(B_poly).exterior.xy, alpha=0.5, color='green')
            elif dist_func.__name__ == 'my_MinWBMP':
                cost = cdist(A, B, metric='euclidean')
                ridx, cidx = linear_sum_assignment(cost)
                for i, j in zip(ridx, cidx):
                    x = np.array([A[i][0], B[j][0]])
                    y = np.array([A[i][1], B[j][1]])
                    ax.plot(x, y, '-ok', fillstyle='none')
                    # Calculate the midpoint
                    mid_x = x.mean()
                    mid_y = y.mean()
                    edge_cost = cost[i][j]
                    angle = np.arctan2(y[1] - y[0], x[1] - x[0]) * 180 / np.pi  # Convert to degrees
                    # Place the text at the midpoint, slightly offset to avoid overlap with the line
                    ax.text(mid_x, mid_y, f'{edge_cost:.2f}', ha='center', va='bottom', rotation=angle, fontsize=7,
                            color='grey')

                ax.plot(*A.T, '.')
                ax.plot(*B.T, '.')
            elif dist_func.__name__ == 'my_hausdorff_distance':

                hd = my_hausdorff_distance(A, B)
                pairwise_dist = cdist(A, B, metric='euclidean')
                from_, to = np.argwhere(pairwise_dist == hd).flatten()[:2]
                x = np.array([A[from_][0], B[to][0]])
                y = np.array([A[from_][1], B[to][1]])
                ax.plot(x, y, '-ok', fillstyle='none')
                ax.text(x.mean(), y.mean(), s=f'{pairwise_dist[from_, to]:.2f}', color='grey', ha='center', va='bottom')
                ax.plot(*A.T, '.')
                ax.plot(*B.T, '.')
            elif dist_func.__name__ == 'my_modified_hausdorff_distance':
                AB_d6 = _my_d_six(A, B)
                BA_d6 = _my_d_six(B, A)
                argmax = np.argmax([AB_d6, BA_d6])
                mhd = [AB_d6, BA_d6][argmax]
                from_ = A if argmax == 0 else B
                to = B if argmax == 0 else A
                pairwise_dist = cdist(from_, to, metric='euclidean')
                from_nearest_neighbors = np.argmin(pairwise_dist, axis=1)
                for i, j in enumerate(from_nearest_neighbors):
                    x = np.array([from_[i][0], to[j][0]])
                    y = np.array([from_[i][1], to[j][1]])
                    ax.plot(x, y, '-ok', fillstyle='none')
                    ax.text(x.mean(), y.mean(), s=f'{pairwise_dist[i, j]:.2f}', color='grey', ha='center', va='bottom')
                ax.plot(*A.T, '.')
                ax.plot(*B.T, '.')

            if col_idx == 0:
                ax.set_ylabel(dist_func.__name__, rotation=90, fontsize=9, ha='center', y=0.5)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    rng = np.random.default_rng(8789)
    depot = (0, 0)
    all_distance_funcs = [my_hausdorff_distance,
                          my_modified_hausdorff_distance,
                          my_convex_hull_jaccard_distance,
                          my_tsp_hull_jaccard_distance,
                          my_tsp_obj_val_diff,
                          my_MinWBMP,
                          ]
    # n_true = 4
    # clients_true = rng.uniform(low=(0, 0), high=(1, 1), size=(n_true, 2))
    # # n_pred = 4
    # # clients_pred = rng.uniform(low=(0, 0), high=(1, 1), size=(n_pred, 2))
    # records = []
    # for scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     clients_pred = clients_true.copy() + rng.normal(0, scale, size=(n_true, 2))
    #     # my_plot(clients_true, clients_pred, depot)
    #     record = {'scale': scale,
    #               'hausdorff': my_hausdorff(clients_true, clients_pred, norm=2),
    #               'degree of intersection': my_jaccard(clients_true, clients_pred),
    #               'MinWBMP': my_MinWBMP(clients_true, clients_pred, norm=2),
    #               }
    #     # print(record)
    #     records.append(record)
    # df = pd.DataFrame(records)
    # fig = px.line(df, x='scale', y=['hausdorff', 'degree of intersection', 'MinWBMP'])
    # fig.show()
    # print(df)

    # ===============================================================

    # df1 = experiment1(100, metrics=all_metrics)
    # scatter = px.scatter_matrix(df1)
    # scatter.show()

    # ===============================================================
    # exp_increasing_add_noise(noise_levels=np.linspace(0, 1, 20), num_samples=100, distances=all_distance_funcs)
    # ===============================================================
    # exp_outlier_sensitivity(all_distance_funcs)
    # ===============================================================
    # num_cases = 1
    # rng = np.random.default_rng(111)
    # n, n_hat = 4, 4
    # cases = []
    # for _ in range(num_cases):
    #     A, B = rng.uniform(0, 25, (n, 2)), rng.uniform(0, 25, (n_hat, 2))
    #     cases.append((A, B))
    # fig = plot_cases(cases, [my_convex_hull_jaccard_distance])
    # plt.show()
    # ===============================================================
    A = np.array([[2, 2], [2, 23], [23, 23], [23, 2]])
    B = np.array([[10, 10], [10, 15], [15, 15], [15, 10]])
    fit = plot_cases([(A, B)], [my_convex_hull_jaccard_distance])
    plt.show()

    pass
