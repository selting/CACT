import sqlite3

import dash_ag_grid as dag
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_daq as daq
import fivecentplots as fcp
import matplotlib  # pip install matplotlib
import pandas as pd  # pip install pandas
from dash import Dash, html, dcc, Input, Output, no_update  # pip install dash

from utility_module.utils import univie_colors_100

matplotlib.use('agg')
import base64

left_col_width = 3


def make_AgGrid_selector(df, col_name: str, width=125):
    row_data = [{col_name: i} for i in df[col_name].unique()]

    return dag.AgGrid(
        id=col_name,
        rowData=row_data,
        columnDefs=[{'field': col_name,
                     'checkboxSelection': True,
                     'headerCheckboxSelection': True,
                     'headerCheckboxSelectionFilteredOnly': True, }],
        defaultColDef={'maxWidth': width},
        dashGridOptions={'pagination': True if len(row_data) > 100 else False,
                         # 'paginationAutoPageSize': True,
                         'rowSelection': 'multiple',
                         'rowHeight': 22,
                         'groupHeaderHeight': 32,
                         'headerHeight': 32,
                         # 'isRowSelectable': {'function': 'log(params)'},
                         # 'isRowSelectable': {'function': f'params.data.{col_name} > 3'},
                         },
        selectedRows=[{col_name: i} for i in df[col_name].unique()],
        # getRowId='params.data.fecha',
        style={'height': 400, 'width': width}
    )


con = sqlite3.connect('data/HPC/output.db')
df = pd.read_sql('SELECT * FROM auctions', con=con)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("Output Evaluation", className='mb-2', style={'textAlign': 'center'}),

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
                value='o',
                clearable=True,
                multi=True,
                options=df.columns[:],
            )
        ], width=8),
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
            daq.BooleanSwitch(id='include-legend',
                              on=True, disabled=False,
                              label='Include legend in groups', ))
    ]),

    # filtering options
    dbc.Row([

        # dbc.Col([
        #     html.Label('n'),
        #     make_AgGrid_selector(df, 'n'),
        # ]),
        # dbc.Col([
        #     html.Label('o'),
        #     make_AgGrid_selector(df, 'o'),
        # ]),
        # dbc.Col([
        #     html.Label('h'),
        #     make_AgGrid_selector(df, 'time_window_length_hours'),
        # ]),
        # dbc.Col([
        #     html.Label('L'),
        #     make_AgGrid_selector(df, 'fin_auction_num_bidding_jobs'),
        # ]),
        dbc.Col([
            html.Label('experiment_id'),
            make_AgGrid_selector(df, 'experiment_id'),
        ]),
        dbc.Col([
            html.Label('fin_auction_rr_search_space'),
            make_AgGrid_selector(df, 'fin_auction_rr_search_space'),
        ]),
        # dbc.Col([
        #     html.Label('fin_auction_rr_search_algorithm'),
        #     make_AgGrid_selector(df, 'fin_auction_rr_search_algorithm')
        # ]),
    ]),

    dbc.Row([

        dbc.Col([
            html.Img(id='fig-matplotlib')
        ], width=6)
    ]),

    # row with the plotly graph and the ag-grid
    # dbc.Row([
    #     dbc.Col([
    #         dcc.Graph(id='fig-plotly', figure={})
    #     ], width=12, md=6),
    #     dbc.Col([
    #         dag.AgGrid(
    #             id='grid',
    #             rowData=df.to_dict("records"),
    #             columnDefs=[{"field": i} for i in df.columns],
    #             columnSize="sizeToFit",
    #         )
    #     ], width=12, md=6),
    # ], className='mt-4'),

])


# Create interactivity between dropdown component and graph
@app.callback(
    Output('fig-matplotlib', 'src'),
    # Output('fig-plotly', 'figure'),
    # Output('grid', 'defaultColDef'),
    Input('y_value', 'value'),
    Input('groups', 'value'),
    Input('facet_col', 'value'),
    Input('legend', 'value'),
    Input('include-legend', 'on'),
    Input('experiment_id', 'selectedRows'),
    Input('fin_auction_rr_search_space', 'selectedRows'),
)
def plot_data(
        selected_yaxis: str,
        selected_groups: list[str],
        selected_facet_col: list[str],
        selected_legend: list[str],
        include_legend: bool,
        selected_experiment_id,
        selected_search_space,
):
    print(f'---\nselected_groups: {selected_groups}, type: {type(selected_groups)}')
    print(f'selected_facet_col: {selected_facet_col}, type: {type(selected_facet_col)}')
    print(f'selected_legend: {selected_legend}, type: {type(selected_legend)}')
    print(f'include_legend: {include_legend}, type: {type(include_legend)}')
    print(f'selected_experiment_id: {selected_experiment_id}, type: {type(selected_experiment_id)}')
    print(f'selected_search_space: {selected_search_space}, type: {type(selected_search_space)}')

    # Filter the dataframe based on the selected experiment_ids
    if not selected_experiment_id:
        return no_update
    if not selected_search_space:
        return no_update
    selected_experiment_id = [x['experiment_id'] for x in selected_experiment_id]
    selected_search_space = [x['fin_auction_rr_search_space'] for x in selected_search_space]

    filtered_df = df.query(
        'experiment_id in @selected_experiment_id and fin_auction_rr_search_space in @selected_search_space')

    # build matplotlib figure with fivecentplots
    selected_groups = [selected_groups] if isinstance(selected_groups, str) else selected_groups
    if bool(selected_legend) and include_legend is True:
        if isinstance(selected_groups, str):
            selected_groups = [selected_groups, selected_legend]
        elif isinstance(selected_groups, list):
            selected_groups += selected_legend

    if selected_legend:
        colors = univie_colors_100[:len(pd.MultiIndex.from_frame(df[selected_legend]).unique())]
    else:
        colors = list(range(len(df[selected_groups].unique())))
    print(f'colors: {colors}')

    style_kwargs = dict(ax_size=[500, 400],
                        # unspecified: rotate long labels, auto: auto-resizes figure, manual: specify size
                        box_stat_line_width=0,
                        marker_edge_color=colors,
                        marker_edge_width=1,
                        marker_edge_alpha=0.6,
                        jitter=0.5,
                        # marker_fill_color=colors,
                        box_fill_color=colors,
                        box_fill_alpha=0.6,
                        box_edge_width=0,
                        box_whisker_color=colors,
                        box_whisker_width=1,
                        box_median_color='#ffffff',
                        box_range_lines=False,  # only whiskers, no range lines

                        save=True,
                        return_filename=True,
                        print_filename=True,
                        filepath='.tmp')

    if selected_groups:
        if selected_legend:
            if selected_facet_col:
                fcp_fig_path = fcp.boxplot(
                    filtered_df,
                    y=selected_yaxis,
                    groups=selected_groups,
                    legend=selected_legend,
                    col=selected_facet_col,
                    **style_kwargs
                )
            else:
                fcp_fig_path = fcp.boxplot(
                    filtered_df,
                    y=selected_yaxis,
                    groups=selected_groups,
                    legend=selected_legend,
                    # col=selected_facet_col,
                    **style_kwargs
                )
        else:
            fcp_fig_path = fcp.boxplot(
                filtered_df,
                y=selected_yaxis,
                groups=selected_groups,
                # col=selected_facet_col,
                **style_kwargs
            )
    else:
        fcp_fig_path = fcp.boxplot(filtered_df, y=selected_yaxis, **style_kwargs)

    fcp_fig_path = fcp_fig_path[5:]
    fcp_fig_data = base64.b64encode(open(fcp_fig_path, 'rb').read()).decode("ascii")
    fig_bar_fcp = f'data:image/png;base64,{fcp_fig_data}'
    fig_matplotlib = fig_bar_fcp

    return fig_matplotlib


if __name__ == '__main__':
    app.run_server(debug=True, port=8002)
