from shiny import reactive, ui, module, Inputs, Outputs

from db_functions import fetch_distinct_param_keys, fetch_distinct_param_values


@module.ui
def filter_row_ui():
    return ui.layout_columns(
        ui.input_selectize('key', '', fetch_distinct_param_keys(), selected=None, multiple=False, width='100%'),
        ui.input_selectize('value', '', [], multiple=True, width='100%'),
        # col_widths=(6, 6)
    )


@module.server
def filter_row_server(input: Inputs, output: Outputs, session):
    @reactive.effect
    def update_value_choices():
        value_choices = fetch_distinct_param_values(input.key())
        ui.update_selectize('value', choices=value_choices)
        pass

    return input.key, input.value
