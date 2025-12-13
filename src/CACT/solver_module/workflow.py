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
    mlflow_experiment = mlflow.set_experiment(experiment_name=f"{instance.id_}")
    # TODO seed management needs an overhaul. Pass the seed/randomstate around or attach it to the instance?
    with mlflow.start_run(
        experiment_id=mlflow_experiment.experiment_id,
        tags={"instance_id": instance.id_, "group_id": mlflow_group_id},
    ):
        seed = generate_unique_seed(
            instance.meta["d"],
            instance.meta["c"],
            instance.meta["n"],
            instance.meta["v"],
            instance.meta["o"],
            instance.meta["r"],
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
            if hasattr(solver, "auction"):
                auction_params = solver.auction.params
                print(
                    "SUCCESS",
                    f"Solved instance {instance}\n"
                    f"with solver:\n"
                    f"{pformat(solver_params)}\n"
                    f"with auction:\n"
                    f"{pformat(auction_params)}\n"
                    f"at {datetime.now()}\n",
                )
            else:
                print(
                    f"SUCCESS"
                    f"Solved instance {instance}\n"
                    f"with solver\n"
                    f"{pformat(solver_params)}\n"
                    f"at {datetime.now()}"
                )

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
