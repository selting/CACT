from datetime import datetime

from solver_module import workflow, config
from utility_module import io
from utility_module.argparse_utils import parser
from importlib.resources import pkg_resources


if __name__ == "__main__":
    print("START main.py")
    # setting args in code
    if True:
        args = {
            "type": "euclidean",
            "distance": 7,
            "num_carriers": 3,
            "num_requests": [10],
            "carrier_max_num_tours": [1],
            "service_area_overlap": [1.0],
            "run": range(1),
            "threads": 1,
            "fail_on_error": 1,
            "tag": "local_dev_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
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

    paths = io.instance_file_selector_2(
        pkg_resources.files("data.instances.euclidean_instances"),
        dict(
            type="euclidean",
            dist_center_to_carrier=[25],
            num_carriers=[3],
            num_requests_per_carrier=[8],
            carriers_max_num_tours=[1],
            carrier_competition=[1],
            run=range(3),
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
