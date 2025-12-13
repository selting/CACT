import sqlite3

import pandas as pd

from utility_module.io import data_dir

replace_dict = {
    # aggregator functions
    'amin': 'min',
    'amax': 'max',
    # misc
    'FitnessRandom': 'Random',
    'FitnessPartitionGanstererHartl': 'GH2018',
    # models
    'LinearRegression': 'LinReg',
    'RidgeRegression': 'Ridge',
    'LassoRegression': 'Lasso',
    'DecisionTreeRegressor': 'DTR',
    'RandomForestRegressor': 'RFR',
    'NeuralNetwork': 'NN',
    'XGBoost': 'XGBR',
    # parameters
    'interaction_degree=1': '',
    'l1_ratio=0.5': '',
    'alpha': '𝛼',
    'output_dim=1': '',
    'lr=0.001': '',  # default
    'hidden_layer_sizes': 'layers',
    'layers=(32,)': 'layers=1',
    'layers=(32, 32, 32)': 'layers=3',
    'activation': 'act',
    'act=ReLU': '',  # default
    'loss_fn=MSELoss': '',
    'num_epochs=100': '',
    'batch_size=16': '',
    'Adam': '',
    'eta': '𝜂',
    '(XGBR)': '(XGBR(𝜂=0.3))',
    # bundle features
    'carrier_specific_feature_num_requests_from_carrier': '# original orders',
    'carrier_specific_feature_num_requests_not_from_carrier': '# non-original orders',
    'carrier_specific_feature_fraction_requests_from_carrier': 'fraction of original orders',
    'carrier_specific_feature_fraction_requests_not_from_carrier': 'fraction of non-original orders',
    'carrier_specific_feature_min_duration_to_carrier_depot': 'min(duration to depot)',
    'carrier_specific_feature_max_duration_to_carrier_depot': 'max(duration to depot)',
    'carrier_specific_feature_sum_duration_to_carrier_depot': 'sum(duration to depot)',
    'carrier_specific_feature_sum_of_squared_duration_to_carrier_depot': 'sum((duration to depot)²)',
    'carrier_specific_feature_mean_duration_to_carrier_depot': 'mean(duration to depot)',
    'carrier_specific_feature_std_duration_to_carrier_depot': 'stdev(duration to depot)',
    'carrier_specific_feature_mean_vertex_carrier_depot_angle': 'mean(angle between customer & depot)',
    'carrier_specific_feature_std_vertex_carrier_depot_angle': 'stdev(angle between customer & depot)',
    'carrier_specific_feature_tour_with_depot_sum_travel_duration': 'TSPTW with depot',

}


# def remove_non_matching_parentheses(s: str):
#     # if the string has the same number of opening and closing parentheses, return the string
#     if s.count('(') == s.count(')'):
#         return s
#
#     # otherwise, find the outermost substring that is enclosed by parentheses
#     pattern = r'(?<=\()(.+)(?=\))'
#
#     matches = []
#     match = re.search(pattern, s).group()
#     while match.count('(') != match.count(')'):
#         matches.append(match)
#         match = re.search(pattern, s).group()
#
#     return s_


def generate_mapping():
    print('reading from database')
    conn = sqlite3.connect(data_dir.joinpath('HPC/output.db'))
    ff_df = pd.read_sql_query("SELECT DISTINCT fin_auction_rr_fitness_function "
                              "FROM auctions ",
                              # "WHERE tag IS NOT NULL",
                              conn)
    conn.close()

    print('cleaning')
    tmp = ff_df['fin_auction_rr_fitness_function'].str.split('(', n=1, expand=True)
    # split into aggregator and feature
    tmp.columns = ['aggr', 'feat']
    tmp['feat'] = tmp['feat'].str[:-1]
    # remove BundleFeature and its parentheses
    mask = tmp['feat'].str.contains('BundleFeature', na=False)
    tmp['feat'][mask] = tmp['feat'][mask].str.replace('BundleFeature(', '', regex=False)
    tmp['feat'][mask] = tmp['feat'][mask].str[:-1]

    # reunite
    print('rejoining columns')
    ff_df['cleaned'] = tmp['aggr'] + '(' + tmp['feat'] + ')'
    ff_df['cleaned'].fillna(tmp['aggr'], inplace=True)

    # replace some strings
    print('replacing name')
    ff_df['replaced'] = ff_df['cleaned']
    for key, value in replace_dict.items():
        ff_df['replaced'] = ff_df['replaced'].str.replace(key, value)

    # remove redundant commas
    print('removing redundant commas')
    ff_df['replaced'] = ff_df['replaced'].str.replace('(, ', '(')
    ff_df['replaced'] = ff_df['replaced'].str.replace(', ( ', '(')
    ff_df['replaced'] = ff_df['replaced'].str.replace(',)', ')')
    ff_df['replaced'] = ff_df['replaced'].str.replace(', , ', ', ')
    for _ in range(5):
        ff_df['replaced'] = ff_df['replaced'].str.replace(', )', ')')

    # # remove redundant parentheses HEURISTICALLY
    print('removing redundant parentheses')
    ff_df['replaced'] = ff_df['replaced'].apply(lambda x: x[:-1] if x.count('(') != x.count(')') else x)
    ff_df['replaced'] = ff_df['replaced'].str.replace('()', '', regex=False)

    ff_df['replaced'] = ff_df['replaced'].str.replace(', )', ')')  # remove redundant commas
    ff_df['replaced'] = ff_df['replaced'].str.replace('(, ', '(')  # remove redundant commas

    print('writing to csv')
    ff_df.to_csv(data_dir.joinpath('fitness_function_mapping.csv'), index=False)
    return ff_df


if __name__ == '__main__':
    ff_df: pd.DataFrame = generate_mapping()
