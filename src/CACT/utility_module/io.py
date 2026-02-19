import datetime as dt
import json
import pathlib
from typing import Dict, List, Any
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from pprint import pprint
from typing import Sequence

import numpy as np
import pandas as pd

from utility_module import utils as ut

import importlib.resources as pkg_resources


# working_dir = Path().cwd()
# # print(working_dir.as_posix())
# data_dir = working_dir.absolute().joinpath("data")
# if not data_dir.exists():
#     raise FileExistsError(
#         f"Data directory does not exist! Make sure the working directory "
#         f"(place from where execution is called) is correct "
#     )
# output_dir = data_dir.joinpath("Output")
# output_dir.mkdir(parents=True, exist_ok=True)
#
# logging_dir = output_dir.joinpath("logs")
# logging_dir.mkdir(parents=True, exist_ok=True)
#
# cr_ahd_instances_dir = data_dir.joinpath("CR_AHD_instances")
# vienna_instances_dir = cr_ahd_instances_dir.joinpath("vienna_instances")
# vienna_train_instance_dir = cr_ahd_instances_dir.joinpath("vienna_train_instances")
# euclidean_instance_dir = cr_ahd_instances_dir.joinpath("euclidean_instances")
# vienna_instance_creation_dir = vienna_instances_dir.joinpath("instance_creation")
#
# cr_ahd_solution_dir = data_dir.joinpath("CR_AHD_solutions")
# cr_ahd_solution_dir.mkdir(parents=True, exist_ok=True)

numeric_columns = [
    "d",
    "c",
    "n",
    "v",
    "o",
    "r",
    # 'tour_improvement_time_limit_per_carrier',
    # 'time_window_length',
    "fin_auction_num_submitted_requests",
    "fin_auction_num_bidding_jobs",
    "fin_auction_num_auction_rounds",
    "objective",
    "sum_travel_distance",
    "sum_travel_duration",
    "sum_wait_duration",
    "sum_service_duration",
    "sum_idle_duration",
    "sum_load",
    "sum_revenue",
    "utilization",
    "num_tours",
    "num_pendulum_tours",
    "num_routing_stops",
    "acceptance_rate",
    "degree_of_reallocation",
    "runtime_final_improvement",
    "runtime_total",
    "runtime_auction",
]


class MyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dt.datetime):
            return o.isoformat()
        if isinstance(o, dt.timedelta):
            return o.total_seconds()
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, frozenset):
            return list(o)
        else:
            return super().default(o)


# class CAHDSolutionSummaryCollection:
#     def __init__(self, solutions: list[Dict]):
#         self.summaries = solutions


def solutions_to_df(solutions, agg_level: str):
    """
    :param solutions: A List of solutions.
    :param agg_level: defines up to which level the solution will be
    summarized/aggregated. E.g. if agg_level='carrier', the returned pd.DataFrame contains infos per carrier but not
    per tour since tours are aggregated for each carrier.
    """
    assert solutions, f"No solutions available"
    df = []
    for solution in solutions:
        if agg_level == "solution":
            record: dict = solution.summary()
            record.pop("carrier_summaries")
            # replace None with np.nan if the feature is numeric
            record = {
                k: (np.nan if (k in numeric_columns and v is None) else v)
                for k, v in record.items()
            }
            df.append(record)

        elif agg_level == "carrier":
            raise NotImplementedError(
                "override of dictionary is not secure yet. E.g. timings will be copied"
            )
            for carrier in solution.carriers:
                record = solution.summary()
                record.pop("carrier_summarCAHDInstance(ies")
                record.update(carrier.summary())
                record.pop("tour_summaries")
                df.append(record)

        elif agg_level == "tour":
            raise NotImplementedError(
                "override of dictionary is not secure yet. E.g. timings will be copied"
            )
            for carrier in solution.carriers:
                for tour in carrier.tours:
                    record = solution.summary()
                    record.pop("carrier_summaries")
                    record.update(carrier.summary())
                    record.pop("tour_summaries")
                    record.update(tour.summary())
                    df.append(record)

        else:
            raise ValueError('agg_level must be one of "solution", "carrier" or "tour"')

    df = pd.DataFrame.from_records(df)

    # convert timedelta to seconds
    df["time_window_length_hours"] = df["time_window_length"].dt.total_seconds() / (
        60**2
    )
    for column in df.select_dtypes(include=["timedelta64"]):
        df[column] = df[column].dt.total_seconds()

    return df


def auctions_results_to_df(auctions):
    assert auctions, f"No auctions available"
    df = []
    for auction in auctions:
        record: dict = auction.summary()
        record.pop("auction_request_pool")
        record.pop("original_assignment")
        record.pop("auction_bundle_pool")
        record.pop("bids_matrix")
        record.pop("winner_assignment")
        record.pop("original_bundle_bids")
        record.pop("winner_bundle_bids")

        # replace None with np.nan if the feature is numeric
        record = {
            k: (np.nan if (k in numeric_columns and v is None) else v)
            for k, v in record.items()
        }
        # convert timedelta to seconds
        record["time_window_length"] = record["time_window_length"].total_seconds()
        # record['bids_on_original_bundles'] = [x.total_seconds() for x in record['bids_on_original_bundles'] if x]
        # record['bids_matrix'] = [[y.total_seconds() for y in x] for x in record['bids_matrix']]
        # record['bids_on_winner_bundles'] = [x.total_seconds() for x in record['bids_on_winner_bundles'] if x]

        df.append(record)

    df = pd.DataFrame.from_records(df)

    # add convenience columns
    for col_name in df.columns:
        if ("duration" in col_name and "rel_" not in col_name) or (
            col_name == "time_window_length"
        ):
            df.insert(
                loc=df.columns.get_loc(col_name) + 1,
                column=col_name + "_hours",
                value=df[col_name] / 60**2,
            )
            df.insert(
                loc=df.columns.get_loc(col_name) + 1,
                column=col_name + "_minutes",
                value=df[col_name] / 60,
            )
    return df


def auctions_bundle_pools_to_df(auctions: Sequence):
    """
    collect all auctions' generated bundles in a df with initial columns indicating the instance.id_ and the solver
    config. further columns are the generated bundles. Some records may be shorter due to fewer bundles


    :param auctions:
    :return:
    """
    df = []
    for auction in auctions:
        auction_bundle_dict = {
            f"bundle_{i:03d}": b for i, b in enumerate(auction.auction_bundle_pool)
        }
        record: dict = dict(
            **auction.meta,
            **auction.solver_config,
            **auction_bundle_dict,
        )
        df.append(record)
    df = pd.DataFrame.from_records(df)
    return df


def unique_path(directory, name_pattern) -> Path:
    """
    construct a unique numbered file name based on a template.
    Example template: file_name + '_#{:03d}' + '.json'

    :param directory: directory which shall be the parent dir of the file
    :param name_pattern: pattern for the file name, with room for a counter
    :return: file path that is unique in the specified directory
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def ask_for_overwrite_permission(path: Path):
    if path.exists():
        permission = input(
            f"Should files and directories that exist at {path} be overwritten?\t[y/n]: "
        )
        if permission == "y":
            return True
        else:
            raise FileExistsError
    else:
        return True


def instance_file_selector(
    type_=None,
    distance=None,
    num_carriers=None,
    num_requests=None,
    carrier_max_num_tours=None,
    service_area_overlap=None,
    run=None,
) -> list[Path]:
    """

    :param type_:
    :param distance: distance of the depots from the city center
    :param num_carriers: number of carriers
    :param num_requests: number of requests per carrier
    :param service_area_overlap: degree of overlap of the service areas between 0 (no overlap) and 1 (all carriers serve the whole city)
    :param run: run, i.e. which of the random instances with the above parameters
    :return: an CAHDInstance
    """
    instance_path = pkg_resources.files("data.instances")

    if type_ == "vienna_train":
        directory = instance_path / "vienna_train_instance_dir"
    elif type_ == "vienna":
        directory = instance_path / "vienna_instances_dir"
    elif type_ == "euclidean":
        directory = instance_path / "euclidean_instances"
    else:
        raise ValueError()

    # TYPE
    if isinstance(type_, str):
        p_type = type_
    elif type_ is None:
        p_type = "\w+"
    elif isinstance(type_, (list, tuple)):
        p_type = "|".join(str(x) + r"\b" for x in type_)
        p_type = f"({p_type})"
    else:
        raise ValueError(
            f"type_ must be str or list of str. type_={type_} is type {type(type_)}"
        )

    # DISTANCE
    if isinstance(distance, int):
        p_distance = distance
    elif distance is None:
        p_distance = "\d+"
    elif isinstance(distance, (list, tuple, range)):
        p_distance = f"({'|'.join((str(x) for x in distance))})"
    else:
        raise ValueError(
            f"distance must be int or list of int. distance={distance} is type {type(distance)}"
        )

    # NUM_CARRIERS
    if isinstance(num_carriers, int):
        p_num_carriers = num_carriers
    elif num_carriers is None:
        p_num_carriers = "\d+"
    elif isinstance(num_carriers, (list, tuple, range)):
        p_num_carriers = f"({'|'.join((str(x) for x in num_carriers))})"
    else:
        raise ValueError(
            f"num_carriers must be int or list of int. num_carriers={num_carriers} is type {type(num_carriers)}"
        )

    # NUM_REQUESTS
    if isinstance(num_requests, int):
        p_num_requests = num_requests
    elif num_requests is None:
        p_num_requests = "\d+"
    elif isinstance(num_requests, (list, tuple, range)):
        p_num_requests = f"({'|'.join((str(x) for x in num_requests))})"
    else:
        raise ValueError(
            f"num_requests must be int or list of int. num_requests={num_requests} is type {type(num_requests)}"
        )

    # CARRIER_MAX_NUM_TOURS
    if isinstance(carrier_max_num_tours, int):
        p_carrier_max_num_tours = carrier_max_num_tours
    elif carrier_max_num_tours is None:
        p_carrier_max_num_tours = "\d+"
    elif isinstance(carrier_max_num_tours, (list, tuple, range)):
        p_carrier_max_num_tours = (
            f"({'|'.join((str(x) for x in carrier_max_num_tours))})"
        )
    else:
        raise ValueError(
            f"carrier_max_num_tours must be int or list of int. carrier_max_num_tours={carrier_max_num_tours} is type {type(carrier_max_num_tours)}"
        )

    # SERVICE_AREA_OVERLAP
    if isinstance(service_area_overlap, float):
        p_service_area_overlap = f"{int(service_area_overlap * 100):03d}"
    elif service_area_overlap is None:
        p_service_area_overlap = "\d+"
    elif isinstance(service_area_overlap, (list, tuple, range)):
        p_service_area_overlap = (
            f"({'|'.join((f'{int(x * 100):03d}' for x in service_area_overlap))})"
        )
    else:
        raise ValueError(
            f"service_area_overlap must be float or list of float. service_area_overlap={service_area_overlap} is type {type(service_area_overlap)}"
        )

    # RUN
    if isinstance(run, int):
        p_run = f"{run:02d}"
    elif run is None:
        p_run = "\d+"
    elif isinstance(run, (list, tuple, range)):
        p_run = f"({'|'.join((f'{x:02d}' for x in run))})"
    else:
        raise ValueError(
            f"run must be int or list of int. run={run} is type {type(run)}"
        )

    pattern = re.compile(
        f"t={p_type}"
        f"\+d={p_distance}"
        f"\+c={p_num_carriers}"
        f"\+n={p_num_requests}"
        f"\+v={p_carrier_max_num_tours}"
        f"\+o={p_service_area_overlap}"
        f"\+r={p_run}(\.json)"
    )
    paths = []
    for file in directory.iterdir():
        # sorted(directory.glob("*.json"), key=ut.natural_sort_key):
        if pattern.match(file.name):
            paths.append(file)
            # print(file.name)
    # sort the list
    paths.sort(key=ut.natural_sort_key)
    if len(paths) == 0:
        raise ValueError(f"No files found in [{directory}] for regex: {pattern}")

    print(f"Selected {len(paths)} instances with the following attributes:")
    pprint(
        dict(
            distance=distance,
            num_carriers=num_carriers,
            num_requests=num_requests,
            carrier_max_num_tours=carrier_max_num_tours,
            service_area_overlap=service_area_overlap,
            run=run,
        ),
        sort_dicts=False,
    )

    return paths


def instance_file_selector_2(
    dir_path: str, filter_criteria: Dict[str, List[Any]]
) -> List[str]:
    """
    Scans a directory for JSON files and filters them based on nested 'desc' values.

    :param dir_path: Path to the directory to scan.
    :param filter_criteria: Dict where keys match 'desc' keys, and values are lists of allowed matches.
    :return: List of paths to matching JSON files.
    """
    base_path = pathlib.Path(dir_path)
    matched_files = []

    # Iterate through all .json files in the directory
    for file_path in base_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure "meta" exists and is a dictionary
            inst_meta = data.get("meta")
            if not isinstance(inst_meta, dict):
                continue

            # Check if all filter criteria are met
            is_match = True
            for key, allowed_values in filter_criteria.items():
                # Get the value from the file's meta dict
                actual_value = inst_meta.get(key)

                # If the key is missing or the value isn't in our allowed list, it's a bust
                if actual_value not in allowed_values:
                    is_match = False
                    break

            if is_match:
                matched_files.append(str(file_path))

        except (json.JSONDecodeError, PermissionError) as e:
            print(f"Skipping {file_path.name}: {e}")
        
    if len(matched_files) == 0:
        raise ValueError("No instance existst for the given filter: \n", filter_criteria)

    return matched_files


# def instance_selector(run=None, rad=None, n=None):
#     """
#     If no arguments are passed a single, random Gansterer&Hartl instance is being solved.

#     :param run:
#     :param rad:
#     :param n:
#     :return:
#     """
#     # print(f'instance selector: run={run}({type(run)}), rad={rad}({type(rad)}), n={n}({type(n)})')
#     if isinstance(run, int):
#         p_run = run
#     elif run is None:
#         p_run = "\d+"
#     elif isinstance(run, (list, tuple, range)):
#         p_run = f"({'|'.join((str(x) for x in run))})"
#     else:
#         raise ValueError(
#             f"run must be int or list of int. run={run} is type {type(run)}"
#         )

#     if isinstance(rad, int):
#         p_rad = rad
#     elif rad is None:
#         p_rad = "\d+"
#     elif isinstance(rad, (list, tuple, range)):
#         p_rad = f"({'|'.join((str(x) for x in rad))})"
#     else:
#         raise ValueError(
#             f"rad must be int or list of int. rad={rad} is type {type(rad)}"
#         )

#     if isinstance(n, int):
#         p_n = n
#     elif n is None:
#         p_n = "\d+"
#     elif isinstance(n, (list, tuple, range)):
#         p_n = f"({'|'.join((str(x) for x in n))})"
#     else:
#         raise ValueError(f"n must be int or list of int. n={n} is type {type(run)}")

#     pattern = re.compile(f"run={p_run}\+dist=200\+rad={p_rad}\+n={p_n}(\.dat)")
#     paths = []
#     for file in sorted(
#         cr_ahd_instances_dir.joinpath("GH_instances").glob("*.dat"),
#         key=ut.natural_sort_key,
#     ):
#         if pattern.match(file.name):
#             paths.append(file)
#             # print(file.name)
#     if len(paths) == 0:
#         raise ValueError
#     return paths


def get_output_counter() -> int:
    """
    Returns a unique id for the output directory.
    :return:
    """
    conn = sqlite3.connect(spo_dir.joinpath("SPO_training_data.db"))
    cursor = conn.cursor()
    # check if table 'counter' exists and create it if not.
    # create the table such that for each new row that is added, the creation datetime is automatically added.
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS counter ("
        "value INTEGER PRIMARY KEY, "
        "created DATETIME DEFAULT CURRENT_TIMESTAMP"
        ")"
    )

    cursor.execute("SELECT MAX(value) FROM counter")
    row = cursor.fetchone()
    if row[0] is None:
        # create a new row with count value 0 and default datetime
        cursor.execute("INSERT INTO counter DEFAULT VALUES")
        conn.commit()
        return 0
    else:
        # create a new row with the counter value of the previous row + 1 and default datetime
        cursor.execute("INSERT INTO counter (value) VALUES (?)", (row[0] + 1,))
        conn.commit()
        return row[0]


def get_database_column_names(database, table):
    assert database.exists()
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # Execute a PRAGMA query to retrieve the table's schema
    cursor.execute(f"PRAGMA table_info({table})")

    # Fetch all the rows returned by the query
    rows = cursor.fetchall()

    # Extract the column names from the rows
    column_names = [row[1] for row in rows]

    conn.close()

    return column_names


# Function to get the last experiment run
def get_last_experiment_run():
    conn = sqlite3.connect("data/output.db")
    query = "SELECT MAX(experiment_id) FROM auctions;"
    result = conn.execute(query).fetchone()[0]
    conn.close()
    return result if result is not None else 0


def get_git_hash():
    try:
        # Run the Git command to get the commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return commit_hash.decode("utf-8")
    except subprocess.CalledProcessError:
        return "Unknown"


def auctions_df_to_db(auctions_df: pd.DataFrame, tag: str = None):
    """
    Write the auction results to the datat/output.db database.
    The database is created if it does not exist yet. The table 'auctions' is created if it does not exist yet.
    The columns of the table are created based on the columns of the DataFrame. If the table already exists, the
    columns of the DataFrame are added to the table if they do not exist yet.

    Parameters
    ----------
    auctions_df
    tag

    Returns
    -------

    """
    auctions_df.rename(columns={"id_": "instance_id"}, inplace=True)
    auctions_df["original_bundles"] = auctions_df["original_bundles"].astype(str)
    auctions_df["winner_bundles"] = auctions_df["winner_bundles"].astype(str)
    auctions_df["timestamp"] = dt.datetime.now()
    auctions_df["git_hash"] = get_git_hash()

    conn = sqlite3.connect("data/output.db")
    cursor = conn.cursor()

    # check whether the tag has already been used and if so, modify it until its unique
    used_tags = pd.read_sql_query("SELECT DISTINCT tag FROM auctions", conn)
    if tag in used_tags.values:
        suffix = 1
        new_tag = tag
        while new_tag in used_tags.values:
            new_tag = tag + f"_{suffix:03d}"
            suffix += 1
        tag = new_tag
    auctions_df["tag"] = tag

    # retrieve the columns that make up the primary key of the table 'auctions'
    cursor.execute("PRAGMA table_info('auctions')")
    res = tuple(cursor.fetchall())
    existing_columns = [row[1] for row in res]
    primary_key_columns = [row[1] for row in res if row[5] > 0]

    # define one experiment BY THE CONFIG VALUES ONLY, i.e. remove the columns that are not part of the config
    for col in [
        "instance_id",
        "experiment_id",
        "timestamp",
        "git_hash",
        "t",
        "d",
        "c",
        "n",
        "v",
        "o",
        "r",
        "original_bundles",
    ]:
        primary_key_columns.remove(col)

    # add any new columns from the group df to the DB table
    new_columns = [
        (col, dtype)
        for col, dtype in zip(auctions_df.columns, auctions_df.dtypes)
        if col not in existing_columns
    ]
    for new_col, dtype in new_columns:
        alter_query = f"ALTER TABLE {'auctions'} ADD COLUMN {new_col} {str(auctions_df[new_col].dtype)};"
        conn.execute(alter_query)
        conn.commit()

    overview = []
    for name, group in auctions_df.groupby(primary_key_columns, dropna=False):
        experiment_id = get_last_experiment_run() + 1
        group["experiment_id"] = experiment_id
        group.to_sql("auctions", conn, if_exists="append", index=False)
        conn.commit()
        overview_record = {k: v for k, v in zip(primary_key_columns, name)}
        overview_record["experiment_id"] = experiment_id
        overview.append(overview_record)
    conn.close()

    overview_df = pd.DataFrame.from_records(overview, index="experiment_id")
    overview_df.dropna(axis="columns", how="all", inplace=True)
    # print ALL columns of the df
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(overview_df.loc[:, (overview_df != overview_df.iloc[0]).any()])
    print(f"Unique tag: {tag}")
