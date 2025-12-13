from datetime import datetime

from solver_module import workflow, config
from utility_module import io
from utility_module.argparse_utils import parser


if __name__ == "__main__":
    print(f"START main.py")
    # setting args in code
    if True:
        args = {
            "type": "euclidean",
            "distance": 7,
            "num_carriers": 3,
            "num_requests": [10],
            "carrier_max_num_tours": [1],
            "service_area_overlap": [1.0],
            "run": range(4),
            "threads": 1,
            "fail_on_error": 1,
            "tag": "local_dev" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
    # else read from terminal parameters
    else:
        args = parser.parse_args().__dict__

    start = datetime.now()

    paths = io.instance_file_selector(
        type_=args["type"],
        distance=args["distance"],
        num_carriers=args["num_carriers"],
        num_requests=args["num_requests"],
        carrier_max_num_tours=args["carrier_max_num_tours"],
        service_area_overlap=args["service_area_overlap"],
        run=args["run"],
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
