import time

import pandas as pd

from sqlalchemy import create_engine

from key_mapping import key_mapping
from value_mapping import value_mapping


def get_db_connection():
    # Create an engine
    engine = create_engine('postgresql+psycopg2://postgres:admin@localhost/mlflow_db')
    # Connect to the database using the engine
    connection = engine.connect()
    return connection


def query_to_df(query):
    conn = get_db_connection()
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        raise e
    finally:
        conn.close()

    return df


def fetch_distinct_param_keys():
    query = "SELECT DISTINCT KEY FROM PARAMS"
    df = query_to_df(query)
    return [""] + sorted(df['key'].tolist())


def fetch_distinct_tag_keys():
    query = "SELECT DISTINCT KEY FROM TAGS"
    tags_df = query_to_df(query)
    return [""] + sorted(tags_df['key'].tolist())


def fetch_distinct_metrics_keys():
    query = "SELECT DISTINCT KEY FROM METRICS"
    metrics_df = query_to_df(query)
    return [""] + sorted(metrics_df['key'].tolist())


def fetch_distinct_param_values(mapped_key):
    if mapped_key == '':
        return []
    reverse_map = {v: k for k, v in key_mapping.items()}
    if mapped_key in reverse_map:
        key = reverse_map[mapped_key]
    else:
        key = mapped_key

    query = f"SELECT DISTINCT VALUE FROM PARAMS WHERE KEY = '{key}'"
    params_values_df = query_to_df(query)
    mapped_df = params_values_df.replace({'value': value_mapping}, inplace=False)  # TODO: is this required?!?!

    return [""] + sorted(mapped_df['value'].tolist())


def fetch_distinct_tag_values(mapped_key):
    if mapped_key == '':
        return []
    reverse_map = {v: k for k, v in key_mapping.items()}
    if mapped_key in reverse_map:
        key = reverse_map[mapped_key]
    else:
        key = mapped_key

    query = f"SELECT DISTINCT VALUE FROM TAGS WHERE KEY = '{key}'"
    tags_values_df = query_to_df(query)
    mapped_df = tags_values_df.replace({'value': value_mapping}, inplace=False)  # TODO: is this required?!?!

    return [""] + sorted(mapped_df['value'].tolist())


def fetch_filtered_run_uuids(tag_filters, param_filters) -> list:
    """
    Searches through the TAGS table to find the run_uuids that satisfy the given tag_filters.
    Then, given those run_uuids, searches through the PARAMS table to find the run_uuids that additionally
     satisfy the given param_filters.
    :param tag_filters:
    :param param_filters:
    :return:
    """
    tags_filter_query = f"SELECT DISTINCT(RUN_UUID) FROM TAGS"
    if tag_filters:
        tags_filter_query += " WHERE"
        for idx, (key, values) in enumerate(tag_filters.items()):
            if isinstance(values, str):
                values = [values]
            where_condition = ', '.join(map(lambda x: f'\'{x}\'', values))
            tags_filter_query += f" (key = '{key}' AND value IN ({where_condition}))"
            if idx < len(tag_filters) - 1:
                tags_filter_query += " OR"
        tags_filter_query += f" GROUP BY run_uuid HAVING COUNT(DISTINCT(key)) = {len(tag_filters)};"

    tag_run_uuids = query_to_df(tags_filter_query)['run_uuid'].tolist()
    print(f'Extracted {len(tag_run_uuids)} run_uuids')

    # these are the child runs that should not be included:
    child_run_uuids_query = """
    SELECT DISTINCT(RUN_UUID) FROM TAGS
    WHERE key = 'mlflow.parentRunId';
    """
    child_run_uuids = query_to_df(child_run_uuids_query)['run_uuid'].tolist()
    print(f'Extracted {len(child_run_uuids)} CHILD run_uuids')
    tag_run_uuids = [run_uuid for run_uuid in tag_run_uuids if run_uuid not in child_run_uuids]
    print(f'Extracted {len(tag_run_uuids)} PARENT run_uuids')

    if not tag_run_uuids:
        return []

    where_condition = ', '.join(map(lambda x: f'\'{x}\'', tag_run_uuids))
    params_filter_query = f"SELECT DISTINCT(RUN_UUID) FROM PARAMS WHERE RUN_UUID IN ({where_condition})"
    if param_filters:
        params_filter_query += " AND"
        for idx, (key, values) in enumerate(param_filters.items()):
            if isinstance(values, str):
                values = [values]
            where_condition = ', '.join(map(lambda x: f'\'{x}\'', values))
            params_filter_query += f" (key = '{key}' AND value IN ({where_condition}))"
            if idx < len(param_filters) - 1:
                params_filter_query += " AND"
        params_filter_query += f" GROUP BY run_uuid HAVING COUNT(DISTINCT(key)) = {len(param_filters)};"
    run_uuids = query_to_df(params_filter_query)['run_uuid'].tolist()
    return run_uuids


def fetch_runs(run_uuids: list):
    """
    Function to fetch specific runs from the RUNS table, given their run_uuids.
    """
    # get the runs' parameters and values
    where_condition = ', '.join(map(lambda x: f'\'{x}\'', run_uuids))

    # join (in SQL) with the runs table to get the experiment_id, name, start_time, end_time, and status
    query = f"""
    SELECT 
        *
    FROM
        RUNS R
    WHERE
        R.RUN_UUID IN ({where_condition})
    ORDER BY
        R.RUN_UUID
    """
    return query_to_df(query)


def fetch_filtered_params(run_uuids: list):
    where_condition = ', '.join(map(lambda x: f'\'{x}\'', run_uuids))

    # join (in SQL) with the runs table to get the experiment_id, name, start_time, end_time, and status
    params_query = f"""
        SELECT
            E.NAME as EXPERIMENT_NAME,
            R.RUN_UUID,
            P.KEY as PARAM_KEY,
            P.VALUE as PARAM_VALUE
--             T.KEY as TAG_KEY,
--             T.VALUE as TAG_VALUE
        FROM
            RUNS R
            JOIN EXPERIMENTS E ON E.EXPERIMENT_ID = R.EXPERIMENT_ID
            JOIN PARAMS P ON R.RUN_UUID = P.RUN_UUID
--             JOIN TAGS T ON R.RUN_UUID = T.RUN_UUID
        WHERE
            R.RUN_UUID IN ({where_condition})
        """
    params_df = query_to_df(params_query)
    # FIXME: include the TAG! group_id is important to know which runs were executed together. treat tags as params
    # params_pivot = pd.pivot_table(params_df,
    #                               index=['experiment_name', 'run_uuid'],
    #                               columns=['param_key'],
    #                               values='param_value',
    #                               aggfunc='first')
    return params_df


def fetch_runs_experiment_tags_params_metrics(run_uuids: list, num_rows):
    """
    Returns all the information about the runs with the given run_uuids. This includes data from the
    RUNS, EXPERIMENTS, TAGS, PARAMS, and METRICS tables.

    :param num_rows:
    :param run_uuids:
    :return:
    """
    where_condition = ', '.join(map(lambda x: f'\'{x}\'', run_uuids))

    query = f"""
    SELECT
        E.NAME as EXPERIMENT_NAME,
        R.RUN_UUID,
        R.START_TIME,
        R.END_TIME,
        R.STATUS,
        T.KEY as TAG_KEY,
        T.VALUE as TAG_VALUE,
        P.KEY as PARAM_KEY,
        P.VALUE as PARAM_VALUE,
        M.KEY as METRICS_KEY,
        M.VALUE as METRICS_VALUE
    FROM
        RUNS R
        JOIN EXPERIMENTS E ON E.EXPERIMENT_ID = R.EXPERIMENT_ID
        JOIN TAGS T ON R.RUN_UUID = T.RUN_UUID
        JOIN PARAMS P ON R.RUN_UUID = P.RUN_UUID
        JOIN METRICS M ON R.RUN_UUID = M.RUN_UUID
    WHERE
        R.RUN_UUID IN ({where_condition})
    """

    if num_rows:
        query += f" LIMIT {num_rows}"

    print(f'Querying the database for the combined information about {len(run_uuids)} runs ...')
    timer_start = time.time()
    df = query_to_df(query)
    timer_end = time.time()
    print(f'Query took {timer_end - timer_start} seconds (df shape: {df.shape})')
    return df


def fetch_filtered_tags(run_uuids: list):
    where_condition = ', '.join(map(lambda x: f'\'{x}\'', run_uuids))

    # join (in SQL) with the runs table to get the experiment_id, name, start_time, end_time, and status
    tags_query = f"""
        SELECT
            R.RUN_UUID,
            T.KEY as TAG_KEY,
            T.VALUE as TAG_VALUE
        FROM
            RUNS R
            JOIN TAGS T ON R.RUN_UUID = T.RUN_UUID
        WHERE
            R.RUN_UUID IN ({where_condition})
        """
    conn = get_db_connection()
    tags_df = pd.read_sql(tags_query, conn)
    conn.close()
    return tags_df


def fetch_filtered_metrics(run_uuids: list):
    """
    Function to fetch filtered data from the database
    """
    where_condition = ', '.join(map(lambda x: f'\'{x}\'', run_uuids))

    metrics_query = f"""
    SELECT
        E.NAME as EXPERIMENT_NAME,
        R.RUN_UUID,
        M.KEY as METRICS_KEY,
        M.VALUE as METRICS_VALUE
    FROM
        RUNS R
        JOIN EXPERIMENTS E ON E.EXPERIMENT_ID = R.EXPERIMENT_ID
        JOIN METRICS M ON R.RUN_UUID = M.RUN_UUID
    WHERE
        R.RUN_UUID IN ({where_condition})
    """

    conn = get_db_connection()
    metrics_df = pd.read_sql(metrics_query, conn)
    conn.close()

    # metrics_pivot = pd.pivot_table(metrics_df,
    #                                index=['experiment_name', 'run_uuid'],
    #                                columns=['metrics_key'],
    #                                values='metrics_value',
    #                                aggfunc='first')
    return metrics_df


def fetch_params_and_metrics(run_uuids: list):
    params = fetch_filtered_params(run_uuids)
    metrics = fetch_filtered_metrics(run_uuids)
    # generate a multi-index on the columns to differentiate between params and metrics and then merge the two df
    # based on the experiment_name and run_uuid
    # params.columns = pd.MultiIndex.from_product([['params'], params.columns])
    # metrics.columns = pd.MultiIndex.from_product([['metrics'], metrics.columns])
    # print(params.head())
    combined = pd.merge(params, metrics, on=['experiment_name', 'run_uuid'], how='outer')
    return combined


def string_to_numeric_or_timedelta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert columns of type 'object' or 'string' to numeric or timedelta.
    This function is useful when reading data from a database where the data types are not preserved.
    Tries to convert the column to numeric, if it fails, tries to convert it to timedelta.
    If both fail, it does nothing.
    :param df:
    :return:
    """
    for col in df.columns:
        # Check if the column is of type 'object' or 'string'
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except ValueError:
                try:
                    df[col] = pd.to_timedelta(df[col], errors='raise')
                except ValueError:
                    pass
    return df
