import argparse
from pathlib import Path
from evaluation_module.data_preparation import parameters_and_metrics
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tag', nargs=1, type=str)


def export_params_and_metrics(tag: str, out_dir: Path = Path('src/CACT/data/experiments/exports')):
    db_url = os.getenv('CACT_DB_URL')
    if db_url is None:
        raise EnvironmentError("CACT_DB_URL environment variable must be defined")
    tag_filters = {"group_id": tag}
    params, metrics = parameters_and_metrics(db_url=db_url, tag_filters=tag_filters, param_filters=None)
    print(f'Getting data from {db_url} with tag {tag}')
    params: pd.DataFrame
    metrics: pd.DataFrame
    params_file = out_dir.joinpath(f"{tag}-params.parquet")
    params.to_parquet(params_file)
    print(f"params written to: {params_file} ")
    metrics_file = out_dir.joinpath(f"{tag}-metrics.parquet")
    metrics.to_parquet(metrics_file)
    print(f"metrics written to: {metrics_file} ")
    pass

if __name__ == "__main__":
    args = parser.parse_args().__dict__
    tag = args['tag']
    export_params_and_metrics(tag)