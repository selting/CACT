import pandas as pd
import re
import numpy as np

import key_mapping, value_mapping
from db_functions import fetch_filtered_run_uuids, query_to_df, string_to_numeric_or_timedelta


def add_grouped_mean_columns(
    df, pattern=r"^(.*?)_(\d+)$", mean_suffix="__mean", inplace=False
):
    """
    Identifies groups of columns in a DataFrame based on a regex pattern
    (typically 'prefix_number') and adds new columns containing the row-wise
    mean for each identified group.

    Args:
        df (pd.DataFrame): The input DataFrame.
        pattern (str, optional): The regex pattern used to identify columns
                                 and capture the prefix. Must capture the
                                 prefix as group 1.
                                 Defaults to r'^(.*?)_(\d+)$'.
        mean_suffix (str, optional): The suffix to append to the prefix for the
                                     new mean column names. Defaults to '_mean'.
        inplace (bool, optional): If True, modify the DataFrame in place and
                                  return None. If False, return a new DataFrame
                                  with the added columns, leaving the original
                                  unchanged. Defaults to False.

    Returns:
        pd.DataFrame or None: Returns a new DataFrame with added mean columns
                              if inplace=False. Returns None if inplace=True.

    Raises:
        TypeError: If the input df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Step 2: Identify Column Prefixes and group column names
    prefixes_to_cols = {}
    compiled_pattern = re.compile(pattern)  # Compile regex for efficiency

    for col in df.columns:
        match = compiled_pattern.match(col)
        if match:
            prefix = match.group(1)  # Get the captured prefix (e.g., 'featA')
            if prefix not in prefixes_to_cols:
                prefixes_to_cols[prefix] = []
            prefixes_to_cols[prefix].append(col)

    # Decide whether to modify in place or return a copy
    if inplace:
        output_df = df  # Work directly on the input DataFrame
    else:
        output_df = df.copy()  # Work on a copy

    print(
        f"Identified groups: {list(prefixes_to_cols.keys())}"
    )  # Optional: Inform user

    # Step 3: Calculate Means and Add New Columns
    for prefix, cols_list in prefixes_to_cols.items():
        if cols_list:  # Check if the list is not empty
            mean_col_name = f"{prefix}{mean_suffix}"
            # Calculate mean using the original df's data for columns
            output_df[mean_col_name] = df[cols_list].mean(
                axis=1, skipna=False
            )  # axis=1 for row-wise mean, don't skip NAs!

    # Return the result based on the inplace argument
    if inplace:
        return None
    else:
        return output_df


def get_multi_carrier_metrics(df, pattern=r"^(.*?)_(\d+)$", inplace=False):
    """
    Identifies groups of columns in a DataFrame based on a regex pattern
    and stacks

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Step 2: Identify Column Prefixes and group column names
    prefixes_to_cols = {}
    multi_carrier_columns = []
    compiled_pattern = re.compile(pattern)  # Compile regex for efficiency

    for col in df.columns:
        match = compiled_pattern.match(col)
        if match:
            prefix = match.group(1)  # Get the captured prefix (e.g., 'featA')
            if prefix not in prefixes_to_cols:
                prefixes_to_cols[prefix] = []
            prefixes_to_cols[prefix].append(col)
            multi_carrier_columns.append(col)

    # Decide whether to modify in place or return a copy
    if inplace:

        output_df = df  # Work directly on the input DataFrame
    else:
        output_df = df.copy()  # Work on a copy

    # TEMPORARY Step 3: select the columns
    output_df = output_df[["run_uuid"] + multi_carrier_columns]

    output_df = output_df.melt(id_vars="run_uuid")
    output_df[["key", "carrier"]] = output_df["key"].str.rsplit("_", expand=True, n=1)
    # output_df['carrier'] = pd.to_numeric(output_df['carrier'])  #optional
    output_df = output_df[["run_uuid", "carrier", "key", "value"]]

    # Return the result based on the inplace argument
    if inplace:
        return None
    else:
        return output_df


def join_columns(df: pd.DataFrame, **kwargs):
    # Create a copy of the original DataFrame
    df_copy = df.copy()

    # Perform the concatenation operations on the copy
    for name, columns in kwargs.items():

        def join_non_empty(row):
            # Get the values from the specified columns for the current row
            values = [str(row[col]) for col in columns]
            # Filter out empty strings (which were originally NaN)
            non_empty_values = [
                val
                for val in values
                if val and not pd.isna(val) and not val in ["nan", "NaN"]
            ]
            # Join the non-empty values with '+'
            return "+".join(non_empty_values)

        df_copy[name] = df_copy.apply(join_non_empty, axis=1)

    # Return the modified copy
    return df_copy


def complete_steps_fast(df: pd.DataFrame, fill_value=np.nan):
    """
    Efficiently fills in missing steps for multi-step metrics up to the maximum step
    for each key using vectorized operations.

    param: df: long dataframe with columns run_uuid, key, value, step
    """
    max_steps_per_key = df.groupby("key")["step"].max()
    # 1. Get all unique run_uuid and key combinations
    run_key_combos = df[["run_uuid", "key"]].drop_duplicates()
    # 2. Create a list of DataFrames, one for each key with its specific max step
    template_dfs = []

    for key, max_step in max_steps_per_key.items():
        # Get all runs for this key
        runs_for_key = run_key_combos[run_key_combos["key"] == key]["run_uuid"].unique()

        # Create all possible combinations for this key
        key_template = pd.DataFrame(
            {
                "run_uuid": np.repeat(runs_for_key, max_step + 1),
                "key": key,
                "step": np.tile(np.arange(max_step + 1), len(runs_for_key)),
            }
        )

        template_dfs.append(key_template)

    template = pd.concat(template_dfs, ignore_index=True)

    # Merge with original data to get values where available
    result = pd.merge(template, df, on=["run_uuid", "key", "step"], how="left")

    # Fill missing values
    result["value"] = result["value"].fillna(fill_value)

    return result


def parameters_and_metrics(sqlite_path, tag_filters: dict, param_filters: dict):
    """
    get the selected runs' parameters and metrics in a long format table
    """
    RUN_UUIDS_CACHE = fetch_filtered_run_uuids(sqlite_path, tag_filters, param_filters)

    where_condition = ", ".join(map(lambda x: f"'{x}'", RUN_UUIDS_CACHE))
    query_runs = f"""
    SELECT
        *
    FROM
        RUNS R
    WHERE R.RUN_UUID IN ({where_condition})
    """
    df_runs = query_to_df(sqlite_path, query_runs)

    query_experiments = f"""
    SELECT
        *
    FROM
        EXPERIMENTS
    """
    df_experiments = query_to_df(sqlite_path, query_experiments)

    query_tags = f"""
    SELECT
        *
    FROM
        TAGS T
    WHERE RUN_UUID IN ({where_condition})
    """
    df_tags = query_to_df(sqlite_path, query_tags)
    df_tags_wide = df_tags.pivot(columns="key", index="run_uuid", values="value")

    query_params = f"""
    SELECT
        *
    FROM
        PARAMS P
    WHERE RUN_UUID IN ({where_condition})
    """
    df_params = query_to_df(sqlite_path, query_params)
    # NOTE sometimes, some params are stored multiple times, i.e. as arrays -> keep only the last entry
    df_params.drop_duplicates(subset=["key", "run_uuid"], keep="last", inplace=True)

    df_params_wide = df_params.pivot(columns="key", index="run_uuid", values="value")

    # merge the dataframes
    PARAMS = pd.merge(df_runs, df_experiments, on="experiment_id")
    PARAMS = pd.merge(PARAMS, df_tags_wide, on="run_uuid")
    PARAMS = pd.merge(PARAMS, df_params_wide, on="run_uuid")

    # replace and rename
    PARAMS = PARAMS.replace(value_mapping.value_mapping)  # will raise in future versions of pandas
    PARAMS = PARAMS.rename(columns=key_mapping.key_mapping)

    # transform to the correct data types
    PARAMS = string_to_numeric_or_timedelta(PARAMS)

    # get the metrics data
    query_metrics = f"""
    SELECT
        *
    FROM
        METRICS M
    WHERE RUN_UUID IN ({where_condition})
    """
    df_metrics = query_to_df(sqlite_path, query_metrics)
    df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"], unit="ms")
    df_metrics = complete_steps_fast(df_metrics)
    df_metrics = df_metrics.replace(
        {"value": {np.finfo(np.float64).max: np.inf}}
    )  # replace max floats with proper np.inf
    # pivot to enable replacing and renaming
    METRICS = df_metrics.pivot(
        columns="key", index=["run_uuid", "step"], values="value"
    ).reset_index()

    # replace and rename
    METRICS = METRICS.replace(value_mapping.value_mapping)
    METRICS = METRICS.rename(columns=key_mapping.key_mapping)

    # transform to the correct data types
    METRICS = string_to_numeric_or_timedelta(METRICS)

    # melt/unpivot to get the long format again
    METRICS = METRICS.melt(id_vars=["run_uuid", "step"])

    return PARAMS, METRICS


def join_columns(df: pd.DataFrame, **kwargs):
    # Create a copy of the original DataFrame
    df_copy = df.copy()

    # Perform the concatenation operations on the copy
    for name, columns in kwargs.items():

        def join_non_empty(row):
            # Get the values from the specified columns for the current row
            values = [str(row[col]) for col in columns]
            # Filter out empty strings (which were originally NaN)
            non_empty_values = [
                val
                for val in values
                if val and not pd.isna(val) and not val in ["nan", "NaN"]
            ]
            # Join the non-empty values with '+'
            return "+".join(non_empty_values)

        df_copy[name] = df_copy.apply(join_non_empty, axis=1)

    # Return the modified copy
    return df_copy
