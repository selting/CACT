from collections import defaultdict

import yaml
from shiny import reactive, ui, module, Inputs, Outputs

from db_functions import fetch_distinct_tag_keys, fetch_distinct_tag_values

with open('shiny_settings.yaml') as f:
    loaded_settings = yaml.safe_load(f)
    tag_filter_loaded_settings = loaded_settings['TAG_FILTER_CACHE']
    tag_filter_loaded_settings = defaultdict(lambda: None, tag_filter_loaded_settings)


@module.ui
def tag_filter_row_ui():
    return ui.layout_columns(
        ui.input_selectize('tag_key', '', fetch_distinct_tag_keys(), multiple=False, width='100%'),
        ui.input_selectize('tag_value', '', [], multiple=True, width='100%'),
        # col_widths=(6, 6)
    )


@module.server
def tag_filter_row_server(input: Inputs, output: Outputs, session):
    @reactive.effect
    def update_tag_value_choices():
        key = input.tag_key()
        tag_value_choices = fetch_distinct_tag_values(key)
        # if the tag key is the group_id, sort in decreasing order
        if key == 'group_id':
            tag_value_choices = sorted(tag_value_choices, reverse=True)
            if tag_filter_loaded_settings[key] is None:
                selected = tag_value_choices[0]
            else:
                selected = tag_filter_loaded_settings[key]

        else:
            selected = tag_filter_loaded_settings[key]

        ui.update_selectize('tag_value', choices=tag_value_choices, selected=selected)
        pass

    return input.tag_key, input.tag_value
