from collections import defaultdict
from datetime import timedelta, datetime
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import plotnine as pt9
import yaml
from shiny import App, reactive, render, ui, Inputs, Outputs
from shinywidgets import render_widget, output_widget

from db_functions import fetch_filtered_run_uuids, query_to_df, \
    string_to_numeric_or_timedelta
from filter_module import filter_row_ui, filter_row_server
from plotly_templates import scientific_1, draft
from tag_filter_module import tag_filter_row_ui, tag_filter_row_server

print(f'registering custom plotly templates')
pio.templates['scientific_1'] = scientific_1
pio.templates['draft'] = draft

print(f'reading cache')
with open('shiny_settings.yaml') as f:
    loaded_settings = yaml.safe_load(f)
    loaded_settings = defaultdict(lambda: None, loaded_settings)

app_ui = ui.page_fluid(
    # fixing a z-axis bug on input_selectize inside card: https://github.com/posit-dev/py-shiny/issues/779
    ui.tags.style(".card { overflow: visible !important; }"),
    ui.tags.style(".card-body { overflow: visible !important; }"),

    ui.panel_title('CRAHD Dashboard'),
    ui.input_dark_mode(mode='light'),

    ui.card(
        ui.layout_columns(
            ui.card('Filter tags', ui.input_action_button('add_tag_filter_button', 'Add Tags Filter')),
            ui.card('Filter Parameters', ui.input_action_button('add_filter_button', 'Add Params Filter')),
        ),
        id='filter_tab'
    ),

    ui.card(ui.accordion(
        ui.accordion_panel('RUNS only', ui.output_data_frame('runs_df')),
        ui.accordion_panel('Tidy',
                           ui.input_selectize('select_tidy_columns', 'Select columns', choices=(), multiple=True, width='100%'),
                           ui.output_data_frame('tidy_df_out'),
                           ui.download_button("download", "Download CSV")),
        open=False)),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_selectize('select_x_axis', 'X Axis', (),
                               selected=loaded_settings['select_x_axis'],
                               multiple=False, width='100%', ),
            ui.input_selectize('select_metrics', 'Metrics', (),  # fetch_distinct_metrics_keys(),
                               selected=None, multiple=False, width='100%', ),
            ui.input_selectize('select_color', 'Color', (),
                               selected=None, multiple=True, width='100%', ),
            ui.input_selectize('select_facet_col', 'Facet Column', (),
                               selected=None, multiple=False, width='100%', ),
            ui.input_selectize('select_facet_row', 'Facet Row', (),
                               selected=None, multiple=False, width='100%', ),
            position='left', open='open', title='Plot/Pivot Options', bg='#f8f8f8'),
        ui.card(
            ui.input_radio_buttons('plotly_type', 'Plotly Type', ['box', 'violin', 'strip', 'histogram'],
                                   inline=True),
            ui.input_switch('plotly_ylog', 'log y axis', ),
            ui.input_selectize('select_template', 'Plotly Template', list(pio.templates),
                               selected='plotly', multiple=False, width='100%', ),
            # ui.input_switch('x_axis_as_string', 'X Axis As String', ),
            output_widget('plotly_out'),
            ui.download_button("download_plot", "Export Plot"),
            # ui.output_plot('plotnine_out'),
            ui.input_radio_buttons('pivot_aggregation', 'Pivot Aggregation',
                                   ['count', 'mean', 'min', 'max', 'std', 'first', 'last'], inline=True),
            ui.output_table('pivot_table_of_plots'),
        ),
    ),
)


def server(input: Inputs, output: Outputs, session):
    # ⏬ FILTERS

    # dictionary of: {module_namespace: (tag_key, tag_value)}
    TAG_FILTER_CACHE = reactive.Value({})
    # dictionary of: {module_namespace: (param_key, param_value)}
    PARAM_FILTER_CACHE = reactive.Value({})

    @reactive.effect
    @reactive.event(input.add_tag_filter_button)
    def add_tag_filter_row():
        tag_filter_count = len(TAG_FILTER_CACHE())
        # TAG_FILTER_COUNT.set(TAG_FILTER_COUNT() + 1)
        module_namespace = f'tag_filter_{tag_filter_count}'
        ui.insert_ui(tag_filter_row_ui(module_namespace), selector='#add_tag_filter_button', where='beforeBegin', )
        tag_filter: tuple[str, Any] = tag_filter_row_server(module_namespace)

        TAG_FILTER_CACHE.set({**TAG_FILTER_CACHE(), module_namespace: tag_filter})
        pass

    @reactive.effect
    @reactive.event(input.add_filter_button)
    def add_param_filter_row():
        param_filter_count = len(PARAM_FILTER_CACHE())
        # PARAM_FILTER_COUNT.set(PARAM_FILTER_COUNT() + 1)
        module_namespace = f'filter_{param_filter_count}'
        ui.insert_ui(filter_row_ui(module_namespace), selector='#add_filter_button', where='beforeBegin', )
        param_filter: tuple[str, Any] = filter_row_server(f'filter_{param_filter_count}')

        PARAM_FILTER_CACHE.set({**PARAM_FILTER_CACHE(), module_namespace: param_filter})
        pass

    # dict of {tag_key: tag_value}
    VALID_TAG_FILTER_CACHE = reactive.Value({})
    # dict of {param_key: param_value}
    VALID_PARAM_FILTER_CACHE = reactive.Value({})

    @reactive.effect
    def update_valid_filters():
        """
        whenever the filters are updated, update the valid filters. valid filters are those that have
        both key and value being non-empty.
        :return:
        """
        valid_tag_filters = {}
        for _, (k, v) in TAG_FILTER_CACHE().items():
            k, v = k(), v()
            if k and v:
                if isinstance(v, (
                        float, int, timedelta, datetime)):  # FIXME this is completely broken because of the mappings
                    v = str(v)
                valid_tag_filters[k] = v

        if valid_tag_filters != VALID_TAG_FILTER_CACHE():
            print('updating valid tag filters')
            VALID_TAG_FILTER_CACHE.set(valid_tag_filters)

        valid_param_filters = {}
        for _, (k, v) in PARAM_FILTER_CACHE().items():
            k, v = k(), v()
            if k and v:
                if isinstance(v, (
                        float, int, timedelta, datetime)):  # FIXME this is completely broken because of the mappings
                    v = str(v)
                valid_param_filters[k] = v

        if valid_param_filters != VALID_PARAM_FILTER_CACHE():
            print('updating valid param filters')
            VALID_PARAM_FILTER_CACHE.set(valid_param_filters)
        pass

    RUN_UUIDS_CACHE = reactive.Value([])

    @reactive.effect
    def update_run_uuids():
        """
        Updates the list of run_uuids based on the current tag filters and param filters.
        :return:
        """

        # if not valid_tag_filters and not valid_param_filters:
        #     return
        run_uuids = fetch_filtered_run_uuids(VALID_TAG_FILTER_CACHE(), VALID_PARAM_FILTER_CACHE())
        print(f'found {len(run_uuids)} runs')
        RUN_UUIDS_CACHE.set(run_uuids)
        pass

    # 📄 DATAFRAME

    @reactive.calc
    def calc_tidy_df():
"""
query the relevant tables and return a tidy dataframe by merging them.
:return:
"""
if len(VALID_TAG_FILTER_CACHE.get()) == 0 and len(VALID_PARAM_FILTER_CACHE.get()) == 0:
    return pd.DataFrame(['No filter set'], columns=['No filter set'])
if len(RUN_UUIDS_CACHE()) == 0:
    return
where_condition = ', '.join(map(lambda x: f'\'{x}\'', RUN_UUIDS_CACHE()))
query_runs = f"""
SELECT 
    *
FROM
    RUNS R
WHERE R.RUN_UUID IN ({where_condition})
"""
df_runs = query_to_df(query_runs)

query_experiments = f"""
SELECT
    *
FROM
    EXPERIMENTS
"""
df_experiments = query_to_df(query_experiments)

query_tags = f"""
SELECT
    *
FROM
    TAGS T
WHERE RUN_UUID IN ({where_condition})
"""
df_tags = query_to_df(query_tags)
df_tags_wide = df_tags.pivot(columns='key', index='run_uuid', values='value')

query_params = f"""
SELECT
    *
FROM
    PARAMS P
WHERE RUN_UUID IN ({where_condition})
"""
df_params = query_to_df(query_params)
# NOTE sometimes, some params are stored multiple times, i.e. as arrays -> keep only the last entry
df_params.drop_duplicates(subset=['key', 'run_uuid'], keep='last', inplace=True)

df_params_wide = df_params.pivot(columns='key', index='run_uuid', values='value')

query_metrics = f"""
SELECT
    *
FROM
    METRICS M
WHERE RUN_UUID IN ({where_condition})
"""
df_metrics = query_to_df(query_metrics)
df_metrics.drop_duplicates(subset=['key', 'run_uuid'], keep='last', inplace=True)
df_metrics_wide = df_metrics.pivot(columns='key', index='run_uuid', values='value')

# merge the dataframes
tidy = pd.merge(df_runs, df_experiments, on='experiment_id')
tidy = pd.merge(tidy, df_tags_wide, on='run_uuid')
tidy = pd.merge(tidy, df_params_wide, on='run_uuid')
tidy = pd.merge(tidy, df_metrics_wide, on='run_uuid')

# transform to the correct data types
tidy = string_to_numeric_or_timedelta(tidy)

# NOTE print unique values for each parameter and metric column. helpful to fill in the key_mapping and
#  value mapping dictionaries

result = {}
for column in tidy.select_dtypes(['object', 'string']).columns:
    if column.startswith('auction__') or column.startswith('data__') or column.startswith('solver__') \
            or column.startswith('solution__'):
        uniques = tidy[column].unique().tolist()
        result[column] = {k: k for k in uniques if k not in [None, '', np.nan, 'nan', 'None']}
pprint(result, sort_dicts=False)

return tidy

    @reactive.effect
    def update_tidy_column_choices():
        """
        Updates the choices for the select input in the tidy dataframe tab.
        :return:
        """
        df = calc_tidy_df()
        if df is None:
            return

        choices = [''] + df.columns.tolist()
        ui.update_select('select_tidy_columns', choices=choices, selected=loaded_settings['select_tidy_columns'])
        pass

    @render.data_frame
    def tidy_df_out():
        """
        Fetches information from the RUNS, EXPERIMENTS, TAGS, PARAMS, and METRICS tables for the run_uuids in the cache.
        Then, it melts the dataframe to a tidy format.
        :return:
        """
        tidy_df = calc_tidy_df()
        # display only the columns that are selected in the select input
        print(f'select_tidy_columns: {input.select_tidy_columns()}---' )
        if input.select_tidy_columns() != ():
            tidy_df = tidy_df[list(input.select_tidy_columns())]
        return tidy_df

    @render.download(filename="tidy_df.csv")
    def download():
        yield calc_tidy_df().to_csv()

    @reactive.effect
    def update_plot_choices():
        """
        Updates the choices for the select inputs in the plot settings tab.
        :return:
        """
        df: pd.DataFrame = calc_tidy_df()
        if df is None:
            return

        choices = [''] + df.columns.tolist()
        for selector in ['select_x_axis',
                         'select_metrics',
                         'select_color',
                         'select_facet_col',
                         'select_facet_row']:
            ui.update_select(selector, choices=choices, selected=loaded_settings[selector])

        pass

    # 📊 PLOTS
    PLOTLY_FIGURE_CACHE = reactive.Value(None)

    @render_widget
    def plotly_out():
        """
        the main plotly plot that aggregates across all instances and displays a boxplot, violin plot, strip plot or
        histogram.
        uses the template that is specified in the plot settings tab.
        :return:
        """
        print(f'plotly_out:')
        print(f'\tcheck tag and param filter caches')
        if len(VALID_TAG_FILTER_CACHE.get()) == 0 and len(VALID_PARAM_FILTER_CACHE.get()) == 0:
            print(f'\ttag or/and param filter cache is empty - return None')
            return None

        print(f'\tparsing inputs')
        px_func = getattr(px, input.plotly_type())
        x = input.select_x_axis() if input.select_x_axis() != '' else None
        if not x:
            print(f'\tno x axis selected - return None')
            return None
        y = input.select_metrics()
        if y in ('', None):
            print(f'\tno y/metric selected - return None')
            return None
        color = input.select_color() if input.select_color() != '' else None
        facet_col = input.select_facet_col() if input.select_facet_col() != '' else None
        facet_row = input.select_facet_row() if input.select_facet_row() != '' else None
        template = input.select_template()

        print(f'\tgetting tidy_df')
        tidy_df: pd.DataFrame = calc_tidy_df()

        # preprocessing
        print(f'\t\t color: {color}')
        if len(color) == 1:
            color = color[0]
        elif len(color) > 1:
            # if multiple colors are selected, we create a new column that concatenates the values of the selected
            # columns. this column will be used as the color column.
            color_column = '+'.join(color)
            print(f'\t\tjoined color column name: {color_column}')
            tidy_df[color_column] = tidy_df[list(color)].astype(str).agg('+'.join, axis=1)
            color = color_column

        print(f'\tGenerating plotly figure')
        fig: go.Figure = px_func(
            tidy_df,
            x=x,
            y=y,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            template=template,
        )
        print(f'\tUpdate plotly figure')
        if px_func == px.histogram:
            fig.update_traces(histfunc='avg')
            fig.update_layout(barmode='group')
            fig.update_yaxes(title_text=f'average of {input.select_metrics()}')
        if px_func == px.box:
            if input.select_facet_col() != '':
                fig.update_layout(boxmode='overlay')
        if input.plotly_ylog():
            fig.update_yaxes(type='log')
        if color == facet_col:
            if fig.layout.annotations:  # Check if there are annotations first
                for annotation in fig.layout.annotations:
                    if annotation.y > 0.99:  # Heuristic: facet titles are usually positioned near the top (y>0.99 in paper coordinates)
                        annotation.text = ""  # Set the text of the annotation to an empty string, effectively removing the title

        PLOTLY_FIGURE_CACHE.set(fig)
        return fig

    @render.download(filename="plotly_plot.pdf")  # Define the filename for download
    def download_plot():
        """
        Handles the PDF download of the Plotly plot.
        """
        current_fig = PLOTLY_FIGURE_CACHE.get()  # Get the current Plotly figure from the cache
        if current_fig is None:
            return None  # Or handle the case where there's no plot to download (optional)
        import io  # Import the 'io' module
        # Use io.BytesIO to create an in-memory buffer for the PDF
        pdf_buffer = io.BytesIO()
        pio.write_image(current_fig, pdf_buffer, format="pdf", engine='kaleido',
                        width=1000, height=333,
                        )  # Write PDF to the buffer
        pdf_bytes = pdf_buffer.getvalue()  # Get the PDF bytes from the buffer
        pdf_buffer.close()  # Close the buffer
        yield pdf_bytes  # Yield the PDF bytes for download

    @render.plot
    def plotnine_out():
        if len(VALID_TAG_FILTER_CACHE.get()) == 0 and len(VALID_PARAM_FILTER_CACHE.get()) == 0:
            return None

        x = input.select_x_axis() if input.select_x_axis() != '' else None
        if not x:
            return None
        y = input.select_metrics()
        if y in ('', None):
            return None
        color = input.select_color() if input.select_color() != '' else None
        facet_col = input.select_facet_col() if input.select_facet_col() != '' else None
        facet_row = input.select_facet_row() if input.select_facet_row() != '' else None

        tidy_df = calc_tidy_df()

        if color:
            aes = pt9.aes(x=f'factor({x})', y=y, color=color, )
        else:
            aes = pt9.aes(x=f'factor({x})', y=y, )

        p = (
                pt9.ggplot(tidy_df, aes, ) +
                pt9.geom_boxplot() +
                pt9.facet_grid(rows=facet_row, cols=facet_col, ) +
                pt9.theme(legend_position='top',
                          figure_size=(12, 8),
                          )
        )

        return p

    @render.table(index=True, border=1)
    def pivot_table_of_plots():
        print(f'pivot_table_of_plots:')
        print(f'\tcheck tag and param filter caches')
        if len(VALID_TAG_FILTER_CACHE.get()) == 0 and len(VALID_PARAM_FILTER_CACHE.get()) == 0:
            print(f'\ttag or/and param filter cache is empty - return None')
            return None

        print(f'\tparsing inputs')
        values = input.select_metrics()
        if values in ('', None):
            print(f'\tno metric selected')
            return None
        index = [x for x in [input.select_facet_row(), input.select_facet_col(), input.select_x_axis()] if x]
        if not index:
            print(f'\tno index selected (facet_row, facet_col, x_axis)')
            return None
        columns = input.select_color() if input.select_color() != '' else None
        if not columns:
            print(f'\tno color selected')
            return None
        agg_func = input.pivot_aggregation()
        print(f'\t\tindex: {index}')
        print(f'\t\tcolumns: {columns}')
        print(f'\t\tagg_func: {agg_func}')

        print(f'\tgetting tidy_df')
        tidy_df: pd.DataFrame = calc_tidy_df()

        # pivot table
        print(f'\tGenerating pivot table')
        pivot_table = pd.pivot_table(
            data=tidy_df,
            values=values,
            index=index,
            columns=columns,
            aggfunc=agg_func,
            fill_value=None,  # not exactly sure what this arg is doing
            dropna=True,  # not exactly sure what this arg is doing
            observed=False,  # not exactly sure what this arg is doing
        )

        return pivot_table

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @reactive.effect
    # def update_base_field_choices():
    #     base_field_choices = [input.select_x_axis(), input.select_color(), input.select_facet_col(),
    #                           input.select_facet_row()]
    #     ui.update_select('select_base_field', choices=[""] + base_field_choices)
    #     pass

    # @reactive.effect
    # @reactive.event(input.select_base_field)
    # def update_base_item_choices():
    #     if not bool(input.select_base_field()):
    #         pass
    #     else:
    #         base_item_choices = calc_tidy_df()[input.select_base_field()].unique()
    #         # print(base_item_choices)
    #         ui.update_select('select_base_item', choices=[""] + [str(x) for x in base_item_choices])
    #         pass


app = App(app_ui, server)
