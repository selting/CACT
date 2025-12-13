import numpy as np


def scale_problem_data_dict(problem_data_dict: dict, round_func: str = 'round'):
    """
    Scale all relevant values by the int_scaling_factor to make them integers.
    This is required because PyVRP only works with integers at the moment.

    :param problem_data_dict:
    :param round_func: the rounding function to use, ('round', 'trunc', 'dimacs', 'exact'
    :return:
    """

    def round(x):
        """round to nearest integer"""
        return np.round(x).astype(int)

    def trunc(x):
        """truncate to integer"""
        return np.trunc(x).astype(int)

    def dimacs(x):
        """scale by 10 and truncate to integer"""
        y = np.trunc(10 * x)
        if isinstance(x, np.ndarray):
            return y.astype(int)
        else:
            return int(y)

    def exact(x):
        """scales by 1000 and round to nearest integer"""
        y = np.round(1_000 * x)
        if isinstance(x, np.ndarray):
            return y.astype(int)
        else:
            return int(y)

    scaling_func = locals()[round_func]

    clients_scaled = []
    for client in problem_data_dict['clients']:
        client_scaled = {k: v for k, v in client.items()}
        for k in ['x', 'y', 'service_duration', 'tw_early', 'tw_late']:  # TODO 'delivery',
            client_scaled[k] = scaling_func(client_scaled[k])
        clients_scaled.append(client_scaled)

    depots_scaled = []
    for depot in problem_data_dict['depots']:
        depot_scaled = {k: v for k, v in depot.items()}
        for k in ['x', 'y']:
            depot_scaled[k] = scaling_func(depot_scaled[k])
        depots_scaled.append(depot_scaled)

    vehicle_types_scaled = []
    for vehicle_type in problem_data_dict['vehicle_types']:
        vehicle_type_scaled = {k: v for k, v in vehicle_type.items()}
        for k in ['capacity', 'tw_early', 'tw_late', 'max_duration', 'max_distance']:
            vehicle_type_scaled[k] = scaling_func(vehicle_type_scaled[k])
        vehicle_types_scaled.append(vehicle_type_scaled)

    distance_matrices_scaled = []
    for distance_matrix in problem_data_dict['distance_matrices']:
        distance_matrix_scaled = scaling_func(distance_matrix)
        distance_matrices_scaled.append(distance_matrix_scaled)

    duration_matrices_scaled = []
    for duration_matrix in problem_data_dict['duration_matrices']:
        duration_matrix_scaled = scaling_func(duration_matrix)
        duration_matrices_scaled.append(duration_matrix_scaled)

    problem_data_dict_scaled = dict(
        clients=clients_scaled,
        depots=depots_scaled,
        vehicle_types=vehicle_types_scaled,
        distance_matrices=distance_matrices_scaled,
        duration_matrices=duration_matrices_scaled
    )
    return problem_data_dict_scaled
