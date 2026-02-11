import datetime as dt
import math
from typing import Union, Optional

import numpy as np
from scipy.spatial.distance import squareform, pdist
from tqdm import trange
from sklearn.datasets import make_blobs

from core_module import instance as it
from core_module.depot import Depot
from core_module.request import Request
from utility_module import io, utils as ut


def generate_euclidean_cr_ahd_instance(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dist_center_to_carrier: float,
    num_carriers: int,
    num_requests_per_carrier: int,
    num_clusters_per_carrier: Optional[int],
    cluster_std: Optional[float],
    carriers_max_num_tours: int,
    carrier_competition: float,
    run: int,
    max_vehicle_load: int,
    max_tour_length: int,
    max_tour_duration,
    requests_revenue: Union[float, int, list[float], list[int]],
    requests_service_duration: Union[dt.timedelta, list[dt.timedelta]],
    requests_load: Union[float, int, list[float], list[int]],
    constant_kmh: float,
    plot=False,
    save=True,
):
    if carrier_competition != 1:
        raise NotImplementedError("Only carrier_competition=1 is supported for now")
    if num_carriers < 2:
        raise ValueError("Must have at least 2 carriers for a CR_AHD instance")

    seed = int(
        num_carriers
        + num_requests_per_carrier
        + carrier_competition * 100
        + run
        + carriers_max_num_tours
    )
    rng = np.random.default_rng(seed)

    # create arrays/lists if load, revenue or service duration are scalars
    if isinstance(requests_load, (float, int)):
        requests_load = [requests_load] * (num_carriers * num_requests_per_carrier)

    if isinstance(requests_revenue, (float, int)):
        requests_revenue = [requests_revenue] * (
            num_carriers * num_requests_per_carrier
        )

    if isinstance(requests_service_duration, dt.timedelta):
        requests_service_duration = [requests_service_duration] * (
            num_carriers * num_requests_per_carrier
        )

    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

    # generate evenly positioned depots around the city center
    depots = []
    degree_angles = 360 / num_carriers
    degree_radians = math.radians(degree_angles)
    for i in range(num_carriers):
        d = Depot(
            label=f"Depot {i}",
            vertex=i,
            x=center_x + dist_center_to_carrier * math.cos(i * degree_radians),
            y=center_y + dist_center_to_carrier * math.sin(i * degree_radians),
            tw_open=ut.EXECUTION_START_TIME,
            tw_close=ut.END_TIME,
        )
        depots.append(d)

    # generate the requests
    requests = []
    for carrier_idx in range(num_carriers):
        disclosure_times = list(
            ut.datetime_range(
                start=ut.ACCEPTANCE_START_TIME,
                stop=ut.EXECUTION_START_TIME,
                num=num_requests_per_carrier,
                endpoint=False,
            )
        )
        carrier_requests = []
        if num_clusters_per_carrier is None:
            x = rng.uniform(x_min, x_max, size=num_requests_per_carrier)
            y = rng.uniform(y_min, y_max, size=num_requests_per_carrier)
        else:
            xy, _ = make_blobs(
                num_requests_per_carrier,
                n_features=2,
                centers=num_clusters_per_carrier,
                cluster_std=cluster_std,
                center_box=(x_min, x_max),
                random_state=seed + carrier_idx,
            )
            x, y = xy[:, 0], xy[:, 1]
        for j in range(num_requests_per_carrier):
            request_idx = carrier_idx * num_requests_per_carrier + j
            request = Request(
                vertex_uid=(len(depots) + carrier_idx * num_requests_per_carrier + j),
                label=f"Request {request_idx}",
                index=request_idx,
                x=x[j],
                y=y[j],
                initial_carrier_assignment=carrier_idx,
                disclosure_time=disclosure_times[j],
                revenue=requests_revenue[request_idx],
                load=requests_load[request_idx],
                service_duration=requests_service_duration[request_idx],
                tw_open=ut.EXECUTION_START_TIME,
                tw_close=ut.END_TIME,
            )
            carrier_requests.append(request)
        requests.extend(carrier_requests)

    # generate the Euclidean distance matrix
    xy_coords = np.array([(r.x, r.y) for r in depots + requests])
    distance_matrix = squareform(pdist(xy_coords))

    # generate the duration matrix in datetime.timedelta format
    duration_matrix = distance_matrix / constant_kmh
    duration_matrix = np.vectorize(lambda x: dt.timedelta(hours=x))(duration_matrix)

    instance = it.CAHDInstance(
        id_=f"t=euclidean"
        f"+d={dist_center_to_carrier}"
        f"+c={num_carriers}"
        f"+n={num_requests_per_carrier:02d}"
        f"+v={carriers_max_num_tours}"
        f"+o={int(carrier_competition * 100):03d}"
        f"+r={run:02d}",
        carriers_max_num_tours=carriers_max_num_tours,
        max_vehicle_load=max_vehicle_load,
        max_tour_distance=max_tour_length,
        max_tour_duration=max_tour_duration,
        requests=requests,
        depots=depots,
        duration_matrix=duration_matrix,
        distance_matrix=distance_matrix,
    )

    if plot:
        instance.plot()
    if save:
        instance.write_json(
            io.data_dir.joinpath(
                "CR_AHD_instances", "euclidean_instances", f"{instance.id_}.json"
            )
        )

    return instance


if __name__ == "__main__":
    for num_requests_per_carrier in [8, 16, 32, 64]:
        for run in trange(
            # 100,
            1,
            desc=f"Generating instances with {num_requests_per_carrier} requests per carrier",
        ):
            generate_euclidean_cr_ahd_instance(
                x_min=0,
                x_max=100,
                y_min=0,
                y_max=100,
                dist_center_to_carrier=25,
                num_carriers=3,
                num_requests_per_carrier=num_requests_per_carrier,
                num_clusters_per_carrier=1,
                cluster_std=3,
                carriers_max_num_tours=1,
                carrier_competition=1,
                run=run,
                max_vehicle_load=999999999,
                max_tour_length=999999999,
                max_tour_duration=999999999,
                requests_revenue=1,
                requests_service_duration=dt.timedelta(minutes=4),
                requests_load=1,
                constant_kmh=30,
                plot=True,
                save=False,
            )
