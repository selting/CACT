import copy
import json
import random

from utility_module import io


def get_path_cost(tw_path, lst):
    if len(tw_path) == 0:
        return 0

    global travel_duration
    cost = travel_duration[lst][tw_path[0]]
    for i in range(1, len(tw_path)):
        cost += travel_duration[tw_path[i - 1]][tw_path[i]]
    return cost


def improve_tw_2_opt(org_tw_path, depot):
    global travel_duration
    path = copy.deepcopy(org_tw_path)
    if len(org_tw_path) == 0:
        return 0, path

    cost = get_path_cost(path, depot)
    if len(org_tw_path) == 1:
        return cost, path

    n = len(path)
    while True:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # try reverse
                ti, tj = i, j
                while ti < tj:
                    path[ti], path[tj] = path[tj], path[ti]
                    ti += 1
                    tj -= 1
                ncost = get_path_cost(path, depot)
                delta = ncost - cost
                if delta < 0:
                    improved = True
                    cost = ncost
                else:
                    # roll back
                    ti, tj = i, j
                    while ti < tj:
                        path[ti], path[tj] = path[tj], path[ti]
                        ti += 1
                        tj -= 1
        if not improved:
            break

    return cost, path


def report_stats(asm):  # calculate statistics of a car assignemnt
    global travel_duration, n_req, n_depot
    assert (len(asm) == n_req)
    assert (all(0 <= asm[i] < n_depot for i in range(n_req)))
    sum_pw_duration = [0] * n_depot  # sum of pairwise duration
    for i in range(n_req):
        for j in range(n_req):
            if i == j:
                continue
            if asm[i] == asm[j]:
                sum_pw_duration[asm[i]] += travel_duration[n_depot + i][n_depot + j]
    avg_pw = sum(sum_pw_duration) / len(sum_pw_duration)
    print(f'sum pairwise duration = {sum_pw_duration}, avg = {avg_pw:,.2f}')

    est_duration = [0] * n_depot
    # estimated travel duration to visit all assigned customers from each depot
    # use 2-opt to get local minima of the path
    for d in range(n_depot):
        path = [d]
        for i in range(n_req):
            if asm[i] == d:
                path.append(n_depot + i)
        est_duration[d] = improve_tw_2_opt(path, d)[0]
    avg_path = sum(est_duration) / len(est_duration)
    print(f'approx shortest duration per depot = {est_duration}, avg = {avg_path:,.2f}')


def verify(data):
    global travel_duration, n_req, n_depot
    travel_duration = data['_travel_duration_matrix']
    n_depot = data['num_carriers']
    n_req = data['num_requests']

    asm = data['request_to_carrier_assignment']
    print('orginal assignment')
    report_stats(asm)

    print('random assignment 1')
    random.shuffle(asm)
    report_stats(asm)

    print('random assignment 2')
    random.shuffle(asm)
    report_stats(asm)
    # print('-' * 40)


if __name__ == '__main__':
    paths = io.instance_file_selector(
        directory=io.data_dir.joinpath('CR_AHD_instances/ml_training_vienna_instances'),
        type_='vienna_train',
        distance=7,
        num_carriers=3,
        num_requests=100,
        carrier_max_num_tours=3,
        service_area_overlap=[1.0],  # <--- Does the student only have the 0 overlap instances?!
        run=range(3)
    )

    for ntest, f in enumerate(paths):
        print(f'Processing {f.name}'.center(80, '='))
        content = f.open('r', encoding='utf-8').read()
        data = json.loads(content)
        verify(data)

        if ntest == 5:
            break
