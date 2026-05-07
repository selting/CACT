import argparse
from pathlib import Path
from evaluation_module.data_preparation import parameters_and_metrics
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tag', nargs=1, type=str)

def get_incremental_file_name(base_filename):
    path = Path(base_filename)
    if not path.exists():
        return path

    stem = path.stem   # e.g., "data"
    suffix = path.suffix # e.g., ".parquet"
    directory = path.parent
    
    counter = 1
    new_path = path
    
    # Keep incrementing until the file does not exist
    while new_path.exists():
        new_path = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return new_path
    

def export_params_and_metrics(tag: str, out_dir: Path = Path('src/CACT/data/experiments/exports')):
    db_url = os.getenv('BACKEND_STORE_URI')
    if db_url is None:
        raise EnvironmentError("BACKEND_STORE_URI environment variable must be defined")
    tag_filters = {"group_id": tag}
    print(f'Getting data from {db_url} with tag {tag}')
    params, metrics = parameters_and_metrics(db_url=db_url, tag_filters=tag_filters, param_filters=None)

    params: pd.DataFrame
    metrics: pd.DataFrame

    params_file = out_dir.joinpath(f"{tag}-params.parquet")
    params_file = get_incremental_file_name(params_file)
    params.to_parquet(params_file)
    print(f"params written to: {params_file} ")

    metrics_file = out_dir.joinpath(f"{tag}-metrics.parquet")
    metrics_file = get_incremental_file_name(metrics_file)
    metrics.to_parquet(metrics_file)
    print(f"metrics written to: {metrics_file} ")
    pass

if __name__ == "__main__":
    args = parser.parse_args().__dict__
    tag = args['tag']
    export_params_and_metrics(tag)