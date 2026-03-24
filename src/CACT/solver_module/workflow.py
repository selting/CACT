import datetime as dt
import itertools
import multiprocessing
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Sequence

import mlflow
from tqdm import tqdm

from core_module.instance import CAHDInstance
from solver_module import solver as slv
from utility_module import profiling as pr
from utility_module.random import set_all_seeds, generate_unique_seed


def execute_jobs(
    paths,
    solvers: Sequence[slv.Solver],
    num_threads: int = 1,
    fail_on_error: bool = False,
    mlflow_group_id: str = None,
):
    if mlflow_group_id in (None, ""):
        mlflow_group_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Solving on {num_threads} thread(s)")
    n_jobs = len(paths) * len(solvers)
    jobs = itertools.product(paths, solvers, [fail_on_error], [mlflow_group_id])
    
    # check if an mlflow tracking uri is already set, if not set it to a local sqlite db

    # mlflow.set_tracking_uri("sqlite:///src/CACT/data/experiments/cact.db")
    # use MLFLOW_TRACKING_URI environment variable if set, otherwise default to local sqlite db
    if mlflow.get_tracking_uri() is None:
        mlflow.set_tracking_uri("sqlite:///src/CACT/data/experiments/cact.db")
        print(
            "No MLflow tracking URI set. Defaulting to local SQLite database at src/CACT/data/experiments/cact.db"
        )
    else:
        print(f"MLflow tracking URI is set to {mlflow.get_tracking_uri()}")
    
    # pre-generate all experiments in mlflow to avoid race conditions when multiple processes try to create the same experiment concurrently
    client = mlflow.MlflowClient()
    for path in paths:
        exp_name = path.stem
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            try:
                client.create_experiment(exp_name)
            except mlflow.MlflowException as e:
                # If another process created it concurrently, ignore the error
                if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e):
                    pass
                else:
                    raise e
    
    # select single or multithreaded solving
    if num_threads == 1:
        solutions_and_auctions = []
        for j in tqdm(list(jobs)):
            solutions_and_auctions.append(_execute_job(*j))
    else:
        # without tqdm bar
        # with multiprocessing.Pool(num_threads) as pool:
        #     solutions_and_auctions = list(pool.starmap(_execute_job, jobs, chunksize=1))
        print(f"Attempting to create Pool with {num_threads} workers")
        with multiprocessing.Pool(num_threads) as pool:
            list(
                tqdm(
                    pool.imap(_execute_job_star, jobs),
                    total=n_jobs,
                    desc="Solving CRAHD instances",
                )
            )

    print(
        f"Finished solving {n_jobs} jobs. Logged to mlflow under group_id {mlflow_group_id}"
    )
    pass


def _execute_job_star(args):
    """workaround to be able to display tqdm bar: https://stackoverflow.com/a/67845088/15467861"""
    _execute_job(*args)
    pass


def _execute_job(
    path: Path,
    solver: slv.Solver,
    fail_on_error: bool,
    mlflow_group_id,
):
    instance = CAHDInstance.from_json(path)

    exp_name = f"{instance.id_}"
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        try:
            exp_id = client.create_experiment(exp_name)
        except mlflow.MlflowException as e:
            # If another process created it concurrently, fetch it
            if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e):
                exp = client.get_experiment_by_name(exp_name)
                if exp is None:
                    raise
                exp_id = exp.experiment_id
            else:
                raise
    else:
        exp_id = exp.experiment_id

    # TODO seed management needs an overhaul. Pass the seed/randomstate around or attach it to the instance?
    with mlflow.start_run(
        experiment_id=exp_id,
        tags={"instance_id": instance.id_, "group_id": mlflow_group_id},
    ):
        seed = generate_unique_seed(
            *list(instance.meta.values()),
            # instance.meta["d"],
            # instance.meta["c"],
            # instance.meta["n"],
            # instance.meta["v"],
            # instance.meta["o"],
            # instance.meta["r"],
            solver.params["time_window_length"],
        )
        set_all_seeds(seed)
        mlflow.log_params({"data__" + k: v for k, v in instance.meta.items()})

        try:
            timer = pr.Timer()
            solver.execute(instance)
            timer.stop()
            mlflow.log_metric("runtime_total", timer.duration)
            solver_params = solver.params
            # if hasattr(solver, "auction"):
            #     auction_params = solver.auction.params
            #     print(
            #         "SUCCESS",
            #         f"Solved instance {instance}\n"
            #         f"with solver:\n"
            #         f"{pformat(solver_params)}\n"
            #         f"with auction:\n"
            #         f"{pformat(auction_params)}\n"
            #         f"at {datetime.now()}\n",
            #     )
            # else:
            #     print(
            #         f"SUCCESS"
            #         f"Solved instance {instance}\n"
            #         f"with solver\n"
            #         f"{pformat(solver_params)}\n"
            #         f"at {datetime.now()}"
            #     )

        except Exception as e:
            solver_params = solver.params
            if hasattr(solver, "auction"):
                auction_params = solver.auction.params
                print(
                    f"{e}\nFailed on instance {instance}\n"
                    f"with solver:\n"
                    f"{pformat(solver_params)}\n"
                    f"with auction:\n"
                    f"{pformat(auction_params)}\n"
                    f"at {datetime.now()}\n{e}"
                )
            else:
                print(
                    f"{e}\nFailed on instance {instance}\n"
                    f"with solver\n"
                    f"{pformat(solver_params)}\n"
                    f"at {datetime.now()}\n{e}"
                )
            if fail_on_error:
                raise e

    pass
