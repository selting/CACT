import pandas as pd

from new_experiments import run
from datetime import datetime
import hashlib

import nlopt
import numpy as np
from tqdm import tqdm


def make_stable(obj):
    """Recursively converts unstable/complex objects into stable, hashable primitives."""
    if isinstance(obj, np.ndarray):
        # Flatten array to bytes, but include shape/dtype to avoid cross-type collisions
        return (obj.shape, obj.dtype.name, obj.tobytes())

    elif isinstance(obj, dict):
        # Force sort keys so {'a': 1, 'b': 2} and {'b': 2, 'a': 1} hash identically
        return tuple((k, make_stable(obj[k])) for k in sorted(obj.keys()))

    elif isinstance(obj, (list, tuple)):
        # Recursively sanitize items inside sequences
        return tuple(make_stable(x) for x in obj)

    elif callable(obj):
        # Extract module and name of functions to strip out volatile memory addresses
        mod = getattr(obj, "__module__", "")
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        return f"{mod}.{name}"

    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    else:
        # Fallback for any other objects (custom classes, etc.)
        return str(obj)


def generate_param_hash(param_dict: dict) -> str:
    """Generates a deterministic 16-character hex hash from a dictionary."""
    # 1. Convert everything to a rock-solid, stable structure
    stable_structure = make_stable(param_dict)

    # 2. Hash the string representation of that structure
    hasher = hashlib.sha256()
    hasher.update(str(stable_structure).encode("utf-8"))

    return hasher.hexdigest()[:16]


if __name__ == "main":
    x_min, x_max, y_min, y_max = 0, 100, 0, 100
    v_base_seed = [1]
    v_true_num_locations = [4]
    v_pred_num_locations = [4]
    v_maxeval = [256]
    v_size_auction_pool = [12]
    v_num_bundles = [64]  # No. of Queries
    v_opt_algorithm = [nlopt.GN_DIRECT_L_RAND]
    num_trials = 200

    logs = []
    total = [
        len(x)
        for x in [
            v_base_seed,
            v_true_num_locations,
            v_pred_num_locations,
            v_maxeval,
            v_size_auction_pool,
            v_num_bundles,
            v_opt_algorithm,
        ]
    ]

    pbar = tqdm(total=np.prod(total) * num_trials)
    for base_seed in v_base_seed:
        for size_auction_pool in v_size_auction_pool:
            for num_bundles in v_num_bundles:
                for true_num_locations in v_true_num_locations:
                    for pred_num_locations in v_pred_num_locations:
                        for opt_algorithm in v_opt_algorithm:
                            for maxeval in v_maxeval:
                                for trial_index in range(num_trials):
                                    params = dict(
                                        x_min=x_min,
                                        x_max=x_max,
                                        y_min=y_min,
                                        y_max=y_max,
                                        size_auction_pool=size_auction_pool,
                                        num_bundles=num_bundles,
                                        true_num_locations=true_num_locations,
                                        pred_num_locations=pred_num_locations,
                                        opt_algorithm=opt_algorithm,
                                        maxeval=maxeval,
                                    )
                                    run_id = generate_param_hash({**params, trial_index:trial_index})
                                    seed = base_seed + int(run_id, 16)
                                    params["seed"] = seed

                                    meta = dict(
                                        start=datetime.now(),
                                        run_id=run_id,
                                        base_seed=base_seed,
                                        trial_index=trial_index,
                                    )

                                    try:
                                        res = run(**params)
                                        meta["status"] = "success"
                                    except Exception as e:
                                        res = dict()
                                        meta["status"] = "failed"
                                        print(e)
                                    finally:
                                        meta["end"] = datetime.now()
                                        run_log = {
                                            **{
                                                f"meta__{k}": v for k, v in meta.items()
                                            },
                                            **{
                                                f"params__{k}": v
                                                for k, v in params.items()
                                            },
                                            **{f"res__{k}": v for k, v in res.items()},
                                        }
                                        logs.append(run_log)
                                        pbar.update()
    pbar.close()

    df = pd.DataFrame.from_records(logs)
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
    df_file_name = f"new_experiments_{datetime.now().isoformat().split('.')[0]}.parquet"
    df.to_parquet(f"data/experiments/new_experiments/{df_file_name}")
