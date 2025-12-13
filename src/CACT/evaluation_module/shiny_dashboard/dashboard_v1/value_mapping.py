from key_mapping import key_mapping

value_mapping = {
    'auction__request_reallocation': {
        'OfferSelection': 'OfferSelection'},
    'auction__request_reallocation__fitness_function': {
        'FitnessAssignmentAggregateBundleFitness': 'FitnessAssignmentAggregateBundleFitness',
        'FitnessRandom': 'FitnessRandom',
        'FitnessPartitionAggregateBundleFitness': 'FitnessPartitionAggregateBundleFitness',
        'FitnessPartitionGanstererHartl({})': 'FitnessPartitionGanstererHartl({})'},
    'auction__request_reallocation__fitness_function__aggr': {
        'MinWBMP': 'MinWBMP',
        'sum': 'sum'},
    'auction__request_reallocation__fitness_function__bundle_fitness': {
        'FitnessBundleRidgeRegression': 'FitnessBundleRidgeRegression',
        'FitnessBundleCarrier': 'FitnessBundleCarrier'},
    'auction__request_reallocation__fitness_function__bundle_fitness__higher_is_better': {
        'True': 'True'},
    'auction__request_reallocation__fitness_function__bundle_fitness__tour_construction': {
        'VRPTWMinTravelDurationInsertion': 'VRPTWMinTravelDurationInsertion'},
    'auction__request_reallocation__fitness_function__higher_is_better': {
        'False': 'False'},
    'auction__request_reallocation__fitness_function_str': {
        'FitnessRandom': 'FitnessRandom',
        '-MinWBMP(RidgeRegression(alpha=0.001, interaction_degree=1))': '-MinWBMP(RR(α=0.001))',
        'FitnessPartitionGanstererHartl({})': 'FitnessPartitionGanstererHartl({})',
        '-sum((FitnessBundleCarrier))': '-sum((FitnessBundleCarrier))'},
    'auction__request_reallocation__next_queries': {
        'NextQueriesAssignmentGeneticAlgorithm': 'NextQueriesAssignmentGeneticAlgorithm',
        'NextQueriesPartitionRandom': 'NextQueriesPartitionRandom',
        'NextQueriesPartitionGeneticAlgorithm': 'NextQueriesPartitionGeneticAlgorithm'},
    'auction__request_reallocation__next_queries__logging': {
        'False': 'False'},
    'auction__request_reallocation__tour_construction': {
        'VRPTWMinTravelDurationInsertion': 'VRPTWMinTravelDurationInsertion'},
    'auction__request_reallocation__tour_improvement': {
        'NoMetaheuristic': 'NoMetaheuristic'},
    'auction__request_reallocation__tour_improvement__neighborhoods': {
        '[NoNeighborhood]': '[NoNeighborhood]'},
    'auction__request_reallocation__tour_improvement__time_limit_per_carrier': {},
    'auction__request_selection': {
        'MarginalTravelDurationProxy': 'MarginalTravelDurationProxy'},
    'auction__tour_construction': {
        'VRPTWMinTravelDurationInsertion': 'VRPTWMinTravelDurationInsertion'},
    'auction__tour_improvement': {
        'NoMetaheuristic': 'NoMetaheuristic'},
    'auction__tour_improvement__neighborhoods': {
        '[NoNeighborhood]': '[NoNeighborhood]'},
    'auction__tour_improvement__time_limit_per_carrier': {},

    'auction__bundling_and_bidding__fitness_function': {
        'FitnessAssignmentAggregateBundleFitness': 'FitnessAssignmentAggregateBundleFitness',
        'FitnessRandom': 'FitnessRandom',
        'FitnessPartitionAggregateBundleFitness': 'FitnessPartitionAggregateBundleFitness',
        'FitnessPartitionGanstererHartl({})': 'GH2018'
    },
    'auction__bundling_and_bidding__fitness_function__bundle_fitness': {
        'BundleFitnessLinearRegression': 'BF-LinearRegression',
    },
    'auction__bundling_and_bidding__fitness_function__bundle_fitness__error_function':
        {'mean_absolute_error': 'MAE',
         'mean_squared_error': 'MSE',
         'root_mean_squared_error': 'RMSE',
         'mean_absolute_percentage_error': 'MAPE',
         },

    'auction__bundling_and_bidding__fitness_function__bundle_fitness__optimization_policy__name':
        {'BayesianOptimizationPolicy': 'Bayesian Optimization',
         'TrulyRandomSearch': 'Random Search',
         },

    'data__t': {
        'vienna_train': 'vienna_train'},
    'solver__request_acceptance': {
        'CustomerAcceptanceBehavior': 'CustomerAcceptanceBehavior'},
    'solver__request_acceptance__overbooking': {
        'Reject': 'Reject'},
    'solver__request_acceptance__request_acceptance_attractiveness': {
        'Accept': 'Accept'},
    'solver__request_acceptance__time_window_offering': {
        'FeasibleTW': 'FeasibleTW'},
    'solver__request_acceptance__time_window_selection': {
        'UnequalPreference': 'UnequalPreference'},
    'solver__solver': {
        'IsolatedSolver': 'IsolatedSolver',
        'CollaborativeSolver': 'CollaborativeSolver',
        'CentralSolverPyVrp': 'CentralSolverPyVrp',
        'CentralSolver': 'CentralSolver'},
    'solver__tour_construction': {
        'VRPTWMinTravelDurationInsertion': 'VRPTWMinTravelDurationInsertion'},
    'solver__tour_improvement': {
        'NoMetaheuristic': 'NoMetaheuristic'},
    'solver__tour_improvement__neighborhoods': {
        '[NoNeighborhood]': '[NoNeighborhood]'},
    'solver__tour_improvement__time_limit_per_carrier': {},

}

# also add the key aliases to this dict
print(f'updating value_mapping...')
value_mapping_update = {}
for key, value in value_mapping.items():
    # print(f'\tcheck key: {key}')
    if key in key_mapping:
        # print(f'\t\tkey found in key_mapping: {key_mapping[key]}')
        value_mapping_update[key_mapping[key]] = value
value_mapping.update(value_mapping_update)
