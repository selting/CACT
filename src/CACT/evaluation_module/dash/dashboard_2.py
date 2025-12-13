import itertools
import sqlite3

import dash_ag_grid as dag
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_daq as daq
import matplotlib  # pip install matplotlib
import pandas as pd  # pip install pandas
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, Input, Output, no_update, callback_context
from dash import dcc
from dash.dcc import Graph
from plotly.subplots import make_subplots

from utility_module.utils import univie_colors_100_60

matplotlib.use('agg')

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def reorder_list(to_sort, ref):
    """
    Reorders a list `foo` according to a partial ordering defined by `ref`.
    Args:
        foo: The list to be reordered.
        ref: The reference list defining the partial ordering.
    Returns:
        A new list with the elements of `foo` reordered according to `ref`.
    """
    # Create a dictionary to map elements in `ref` to their indices in `foo`.
    ref_idx = {elem: i for i, elem in enumerate(ref)}
    j = len(ref_idx)
    for elem in to_sort:
        if elem not in ref_idx:
            ref_idx[elem] = j
        j += 1
    # Sort `foo` by the order of elements in `ref`, then by their original order in `foo`.
    return sorted(to_sort, key=lambda x: ref_idx[x])


def get_label(col_or_col_name):
    """
    Returns a human-readable label for the column name.
    If a complete column, i.e. a pd.Series is passed, rename that series.

    :param col_or_col_name:
    :return:
    """
    label_mapping = {
        'n': 'No. of requests per carrier',
        'o': 'Overlap',
        'time_window_length_hours': 'Time window length [h]',
        'fin_auction_rr_search_space': 'Search space',
        'fin_auction_rr_search_algorithm': 'Search algorithm',
        'fin_auction_rr_fitness_function': 'Fitness function',
        'fin_auction_num_bidding_jobs': 'No. of bidding jobs $L$',
        'rel_savings_sum_travel_duration': '$RCG$',
        'rel_query_efficiency': '$QE$',
        'FitnessRandom': 'Random',
        'FitnessPartitionGanstererHartl': 'Gansterer & Hartl',
    }

    if isinstance(col_or_col_name, str):
        return label_mapping.get(col_or_col_name, col_or_col_name)
    elif isinstance(col_or_col_name, pd.Series):
        series: pd.Series = col_or_col_name
        series.name = label_mapping.get(series.name, series.name)
        return series
    elif col_or_col_name is None:
        return None
    else:
        raise ValueError(f'col_or_col_name is of type {type(col_or_col_name)}')


left_col_width = 3
pinned_columns = {'n',
                  'o',
                  'time_window_length_hours',
                  'fin_auction_num_bidding_jobs',
                  'fin_auction_rr_search_space',
                  'fin_auction_rr_search_algorithm',
                  'fin_auction_rr_fitness_function'}
category_orders = {
    'n': None,
    'o': None,
    'time_window_length_hours': None,
    'fin_auction_rr_search_space': ['Bundles', 'Partitions',], # 'Assignments'],
    'fin_auction_rr_search_algorithm': None,
    'fin_auction_rr_fitness_function': ['FitnessRandom', 'FitnessPartitionGanstererHartl'],
    'fin_auction_num_bidding_jobs': None,
}

con = sqlite3.connect('data/HPC/output.db')
df: pd.DataFrame = pd.read_sql('SELECT * FROM auctions', con=con)

def preprocess_categoricals(df):
    for col, cat_order in category_orders.items():
        categories = list(df[col].unique())
        if cat_order:
            categories = reorder_list(categories, cat_order)
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True)
    return df

df = preprocess_categoricals(df)


# for name, group in df.groupby('fin_auction_rr_search_space'):
#     print(name)


def make_AgGrid_selector(df, col_name: str, width=125, height=400, ):
    if df[col_name].dtype.name == 'category':
        row_data = [{col_name: i} for i in df[col_name].cat.categories]
    else:
        row_data = [{col_name: i} for i in df[col_name].unique()]

    return dag.AgGrid(
        id=col_name,
        rowData=row_data,
        columnDefs=[{'field': col_name,
                     'checkboxSelection': True,
                     'headerCheckboxSelection': True,
                     'headerCheckboxSelectionFilteredOnly': True,
                     'flex': 1,
                     'wrapHeaderText': True,
                     'autoHeaderHeight': True
                     }],
        # defaultColDef={'maxWidth': width},
        dashGridOptions={'pagination': True if len(row_data) > 100 else False,
                         # 'paginationAutoPageSize': True,
                         'rowSelection': 'multiple',
                         'rowHeight': 22,
                         # 'groupHeaderHeight': 32,
                         # 'headerHeight': 32,
                         # 'isRowSelectable': {'function': 'log(params)'},
                         # 'isRowSelectable': {'function': f'params.data.{col_name} > 3'},
                         },
        selectedRows=[{col_name: i} for i in df[col_name].unique()],
        persistence=True,
        # getRowId='params.data.fecha',
        style={'height': height, 'width': width}
    )


app.layout = dbc.Container([
    dcc.Store(id='my_store',
              storage_type='session'
              ),
    html.H1("Output Evaluation (dashboard_2)", className='mb-2', style={'textAlign': 'center'}),

    # dropdowns
    dbc.Row([
        # column with text description of the dropdown:
        dbc.Col(dbc.Container([html.P("Select the y-axis value", ), ]), width=left_col_width),

        dbc.Col(
            dcc.Dropdown(
                id='y_value',
                searchable=True,
                value='rel_savings_sum_travel_duration',
                clearable=True,
                multi=False,
                options=df.columns[:],
            ),
            width=4),
    ]),
    dbc.Row([
        dbc.Col(dbc.Container([html.P("Select the group value(s)", ), ]), width=left_col_width),
        dbc.Col([
            dcc.Dropdown(
                id='groups',
                searchable=True,
                value=['fin_auction_num_bidding_jobs'],
                clearable=True,
                multi=True,
                options=df.columns[:],
            )
        ], width=8),
    ]),

    dbc.Row([
        dbc.Col(dbc.Container([html.Label("Select the facet row value(s)", ), ]), width=left_col_width),
        dbc.Col([
            dcc.Dropdown(
                id='facet_row',
                searchable=True,
                value='o',
                clearable=True,
                multi=False,
                options=df.columns[:],
            )
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col(dbc.Container([html.Label("Select the facet column value(s)", ), ]), width=left_col_width),
        dbc.Col([
            dcc.Dropdown(
                id='facet_col',
                searchable=True,
                value='n',
                clearable=True,
                multi=False,
                options=df.columns[:],
            )
        ], width=4),
    ]),

    dbc.Row([
        dbc.Col(dbc.Container([html.P("Select the legend value(s)", ), ]), width=left_col_width),
        dbc.Col([
            dcc.Dropdown(
                id='legend',
                searchable=True,
                value=['fin_auction_rr_search_space'],
                clearable=True,
                multi=True,
                options=df.columns[:],
            )
        ], width=4),
        dbc.Col(
            daq.BooleanSwitch(id='include_legend',
                              on=True, disabled=False,
                              label='Include legend in groups', ))
    ]),

    # filtering options
    dbc.Row([
        dbc.Col([
            html.Label('n'),
            make_AgGrid_selector(df, 'n', height=150),

            html.Label('o'),
            make_AgGrid_selector(df, 'o', height=200),
        ]),
        dbc.Col([
            html.Label('h'),
            make_AgGrid_selector(df, 'time_window_length_hours'),
        ]),
        dbc.Col([
            html.Label('experiment_id'),
            make_AgGrid_selector(df, 'experiment_id', width=400),
        ]),
        dbc.Col([
            html.Label('L'),
            make_AgGrid_selector(df, 'fin_auction_num_bidding_jobs'),
        ]),
        dbc.Col([
            html.Label('fin_auction_rr_search_space'),
            make_AgGrid_selector(df, 'fin_auction_rr_search_space'),
        ]),
        dbc.Col([
            html.Label('fin_auction_rr_search_algorithm'),
            make_AgGrid_selector(df, 'fin_auction_rr_search_algorithm', width=300),
        ]),
        dbc.Col([
            html.Label('fin_auction_rr_fitness_function'),
            make_AgGrid_selector(df, 'fin_auction_rr_fitness_function', width=500),
        ]),
    ]),

    # show the matplotlib graph
    # dbc.Row([
    #     dbc.Col([
    #         html.Img(id='fig-matplotlib')
    #     ]),
    # ]),

    # show the plotly graph
    dbc.Row([dbc.Col([dcc.Graph(id='fig-plotly', mathjax=True)])]),

    dbc.Row([html.Button(id='save-button', children='Save Figure'),
             html.Div(id='save-message', children=''), ]),

    # show the data
    dbc.Row([html.H3('Filtered Data')]),
    dbc.Row([dag.AgGrid(
        id='filtered_df',
        columnDefs=[{'field': c,
                     'headerName': c.replace('_', ' ').lower(),
                     'resizable': True,
                     'pinned': 'left' if c in pinned_columns else False,
                     'sortable': True,
                     # 'filter'
                     } for c in df.columns],
        defaultColDef={
            # "initialWidth": 200,
            "wrapHeaderText": True,
            # "autoHeaderHeight": True,
        },
        rowData=df.to_dict('records'),
        columnSize='autoSize',
        columnSizeOptions={'skipHeader': True},
        # persistence=True,
        style={'height': 800, 'width': '100%'}
    )
    ]),

    # show the content of the store
    dbc.Row([
        dbc.Col([
            html.Pre(id='store-content')
        ])
    ]),
],
    fluid=True,
    # style={'margin': '0', 'width': '100%', }
)


@app.callback(
    Output('store-content', 'children'),
    Input('my_store', 'data')
)
def show_store_content(data):
    return html.Div([
        html.Div(f'{x}: {data[x]}', style={'white-space': 'pre'}) for x in data
    ])


@app.callback(
    # output of the callback are the selectable options
    Output('my_store', 'data'),
    # input of the callback is the selected rows of the AgGrids
    Input('n', 'selectedRows'),
    Input('o', 'selectedRows'),
    Input('time_window_length_hours', 'selectedRows'),
    Input('experiment_id', 'selectedRows'),
    Input('fin_auction_num_bidding_jobs', 'selectedRows'),
    Input('fin_auction_rr_search_space', 'selectedRows'),
    Input('fin_auction_rr_search_algorithm', 'selectedRows'),
    Input('fin_auction_rr_fitness_function', 'selectedRows'),

)
def update_store(*selected):
    # print(f'\nselected: {selected}')
    ctx = callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # print(f'input_id: {input_id}')
    else:
        return no_update

    selected_values = {}
    for column_entries in selected:
        # print(f'column: {column_entries}')
        foo = []
        for entry in column_entries:
            for key, value in entry.items():  # is always just going to be one iteration
                foo.append(value)
        selected_values[key] = foo
    # print(f'selected_values: {selected_values}')

    query = ' and '.join([f'{key} in {value}' for key, value in selected_values.items()])
    filtered_df = df.query(query)
    # print(f'filtered_df: {filtered_df}')

    # retrieve the unique values that are still in the filtered df
    selectable_values = {}
    for col in selected_values.keys():
        if col != input_id:
            selectable_values[col] = filtered_df[col].unique().tolist()
        else:
            selectable_values[col] = df[col].unique().tolist()

    new_store = dict(selected_values=selected_values, selectable_values=selectable_values)

    return new_store


# @app.callback(
#     Output('experiment_id', 'rowData'),
#     Input('my_store', 'data')
# )
# def update_selectable_rows(data):
#     selectable_values = data["selectable_values"]["experiment_id"]
#     # print(f'selectable_values: {selectable_values}')
#     # row_data = [{'experiment_id': i} for i in df['experiment_id'].unique()]
#     return [{'experiment_id': i} for i in selectable_values]
#     # return {'pagination': True if len(row_data) > 100 else False,
#     #         # 'paginationAutoPageSize': True,
#     #         'rowSelection': 'multiple',
#     #         'rowHeight': 44,
#     #         'groupHeaderHeight': 32,
#     #         'headerHeight': 32,
#     #         # 'isRowSelectable': {'function': 'log(params)'},
#     #         # 'isRowSelectable': {'function': f'params.data.experiment_id >= 0 && params.data.experiment_id <= 5'},
#     #         'isRowSelectable': {'function': f'params.data.experiment_id < 3'},
#     #         }


@app.callback(
    Output('filtered_df', 'rowData'),
    Input('my_store', 'data')
)
def update_filtered_df(data):
    selected_values = data["selected_values"]
    query = ' and '.join([f'{key} in {value}' for key, value in selected_values.items()])
    filtered_df = df.query(query)
    return filtered_df.to_dict('records')


def custom_groupby(df, groupby):
    if not groupby:
        yield None, df
    else:
        if len(groupby) == 1:
            groupby = groupby[0]
        for name, group in df.groupby(groupby, observed=True, sort=True):
            yield name, group


# def custom_groupby2(data, groupby, predefined_groups=None):
#     if not groupby:
#         return [(None, data)]
#     else:
#         grouped_df = data.groupby(
#             pd.Categorical(df[groupby].unique(), categories=predefined_groups),
#             observed=False,
#             sort=False)
#         return grouped_df


@app.callback(
    Output('fig-plotly', 'figure'),
    Input('filtered_df', 'rowData'),
    Input('y_value', 'value'),
    Input('groups', 'value'),
    Input('facet_col', 'value'),
    Input('facet_row', 'value'),
    Input('legend', 'value'),
)
def update_plotly_fig(row_data, selected_yaxis, selected_groups, selected_facet_col, selected_facet_row,
                      selected_legend):
    if not row_data:
        return px.bar(pd.DataFrame(columns=df.columns), y='r')
    filtered_df = pd.DataFrame(row_data)
    filtered_df = preprocess_categoricals(filtered_df)

    # get rows and cols and make sure that in case they are of type category, they are sorted according to the category
    if selected_facet_row:
        rows = list(filtered_df[selected_facet_row].unique())
        rows = sorted(rows, key=lambda x: filtered_df[selected_facet_row].cat.categories.get_loc(x))
    else:
        rows = []

    if selected_facet_col:
        cols = list(filtered_df[selected_facet_col].unique())
        cols = sorted(cols, key=lambda x: filtered_df[selected_facet_col].cat.categories.get_loc(x))
    else:
        cols = []

    if selected_groups:
        groups = list(filtered_df[selected_groups[0]].unique())
        groups = sorted(groups, key=lambda x: filtered_df[selected_groups[0]].cat.categories.get_loc(x))
    else:
        groups = []

    # get the colors for the legend
    if selected_legend:
        colors = {legend: color for legend, color
                  in zip(filtered_df[selected_legend[0]].cat.categories, itertools.cycle(univie_colors_100_60))}
    else:
        colors = {None: univie_colors_100_60[0]}
    # pprint(colors)

    fig = make_subplots(rows=len(rows) if len(rows) > 0 else 1,
                        cols=len(cols) if len(cols) > 0 else 1,
                        shared_xaxes='all',
                        shared_yaxes='all',
                        column_titles=[str(x) for x in cols],
                        row_titles=[get_label(str(x)) for x in rows],
                        x_title=get_label(selected_groups[0] if selected_groups else None),
                        y_title=get_label(selected_yaxis),
                        )
    # print(f'\nselected_groups: {selected_groups}')
    # print(f'x = {filtered_df[selected_groups[0]] if selected_groups else None}')
    legend_entries = set()
    for row_idx, (row_name, row_df) in enumerate(custom_groupby(filtered_df, selected_facet_row)):
        print(f'\nrow_name: {row_name}')
        for col_idx, (col_name, row_col_df) in enumerate(custom_groupby(row_df, selected_facet_col)):
            print(f'\tcol_name: {col_name}')
            for leg_rank, (legend_entry, row_col_legend_df) in enumerate(custom_groupby(row_col_df, selected_legend)):
                print(f'\t\tlegend_entry: {legend_entry}')
                x = row_col_legend_df[selected_groups[0]] if selected_groups else None
                y = row_col_legend_df[selected_yaxis]
                showlegend = True if legend_entry not in legend_entries else False
                legend_entries.add(legend_entry)

                fig.add_trace(go.Box(
                    y=get_label(y),
                    x=get_label(x),
                    name=get_label(legend_entry),
                    legendgroup=get_label(legend_entry),
                    showlegend=showlegend,
                    legendrank=leg_rank,
                    marker_color=colors[legend_entry],
                    offsetgroup=get_label(legend_entry),
                ), row=row_idx + 1, col=col_idx + 1)
    fig.update_layout(
        boxmode='group',
        # size
        # width=1000,
        height=800,
        # limit the legend width
        legend=dict(
            entrywidth=0.1,
            entrywidthmode='fraction',
            itemwidth=60,
        ),
        template='plotly_white',
    )
    fig.update_xaxes(showgrid=True, ticks='outside', tickson='boundaries', )  # FIXME tickson='boundaries' does not work
    return fig


@app.callback(
    Output("save-message", "children"),
    Input("save-button", "n_clicks"),
)
def save_figure(n_clicks):
    if n_clicks:
        app
        # fig: dcc.Graph = app.layout['fig-plotly']
        # print(fig)
        # go_fig = go.Figure(fig)
        # go_fig.write_image('fig1.png')

        # Optional: Show a success message (replace with your message)
        return "Figure saved successfully!"
    else:
        return None  # Return nothing if button not clicked


if __name__ == '__main__':
    app.run_server(debug=True, port=8003)
