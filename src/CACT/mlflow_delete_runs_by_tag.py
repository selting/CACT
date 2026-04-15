import mlflow
import subprocess
import os
import argparse
from evaluation_module.db_functions import fetch_filtered_run_uuids
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--tag', nargs=1, type=str)
parser.add_argument('--dryrun', action='store_true')
def check_whether_exported(tag:str):
    # check whether the runs have been exported to make sure that no accidental deletion happens
    metrics_exists = os.path.exists(f"src/CACT/data/experiments/exports/['{tag}']-metrics.parquet")
    if not metrics_exists:
        print(f"No metrics export found for tag {tag}")
        return False
    params_exists = os.path.exists(f"src/CACT/data/experiments/exports/['{tag}']-params.parquet")
    if not params_exists:
        print(f"No params export found for tag {tag}")
        return False
    if metrics_exists and params_exists:
        return True

def delete_runs_by_tag(tag: str, dry_run: bool = False):
    """
    Set runs with the given tag to 'deleted' lifecycle stage
    """
    BACKEND_STORE_URI = os.getenv('BACKEND_STORE_URI')
    if not BACKEND_STORE_URI:
        raise EnvironmentError("BACKEND_STORE_URI environment variable must be defined")
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    if not MLFLOW_TRACKING_URI:
        raise EnvironmentError("MLFLOW_TRACKING_URI environment variable must be defined")
    # Get all runs with the specified tag
    run_uuids = fetch_filtered_run_uuids(db_url=BACKEND_STORE_URI, tag_filters={"group_id": [tag]}, param_filters={})
    # Set them to status "deleted"
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)
    for run_uuid in tqdm(run_uuids, desc=f"Setting runs to 'deleted' in mlflow {'(dry-run)' if dry_run else ''}:"):
        if not dry_run:
            client.delete_run(run_uuid)
    # use mlflow CLI command 'mlflow gc' to permanently delete the runs from the database
    if dry_run:
        print("Dry run mode: No runs will be deleted from the database")
    else:
        print("running mlflow gc to fully delete the runs from the database. Are you sure [y/n]?")
        if input().lower() != 'y':
            print("Aborting garbage collection")
            return
        result = subprocess.run(['mlflow', 'gc'], capture_output=True)
        # Get the output of the command
        output = result.stdout.decode()
        # Print the output
        print(output)



if __name__ == "__main__":
    args = parser.parse_args()
    tag = args.tag[0]
    dry_run = args.dryrun[0]
    if dry_run:
        print("DRY RUN")

    exported = check_whether_exported(tag)

    if not exported:
        # ask user whether runs should be deleted anyway
        print(f"Warning: No (full) export of runs with tag '{tag}' found. Are you sure you want to delete these runs? (y/n)")
        if input().lower() != 'y':
            print("Aborting deletion.")
            exit(1)
        else:
            print("Proceeding with deletion.")
            delete_runs_by_tag(tag, dry_run=dry_run)
    else:
        print(f"Export of runs with tag '{tag}' found. Proceeding with deletion.")
        delete_runs_by_tag(tag, dry_run=dry_run)
