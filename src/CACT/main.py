from datetime import datetime
from pathlib import Path
import sys

from solver_module import workflow, config
from utility_module import io
from utility_module.argparse_utils import parser

if __name__ == "__main__":
    print("START main.py")
    # read command line arguments
    args = parser.parse_args().__dict__
    
    start = datetime.now()

    # TODO include these instance selection parameters into the argument parser
    paths = io.instance_file_selector_2(
        Path("src/CACT/data/instances/euclidean_instances_rev1"),
        dict(
            type="euclidean",
            dist_center_to_carrier=[25],
            num_carriers=[3],
            num_requests_per_carrier=[8],
            carriers_max_num_tours=[1],
            carrier_competition=[1],
            run=range(20),
            num_clusters_per_carrier=[None, 3],
            cluster_std=[None, 3],
        ),
    )

    # print(paths)
    # exit(1)
    solvers = list(config.configs())

    # SOLVING
    workflow.execute_jobs(
        paths=paths,
        solvers=solvers,
        num_threads=args["threads"],
        fail_on_error=args["fail_on_error"],
        mlflow_group_id=args.get("tag"),
    )

    end = datetime.now()
