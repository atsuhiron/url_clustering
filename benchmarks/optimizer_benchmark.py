import copy
import time

import numpy as np
import faker
import tqdm
import matplotlib.pyplot as plt

from query_parsing import parsed_queries, parser
from vo.reconst_coord import ReconstCoord
from fake_query_generator.fq_generator import FQGenerator
from fake_query_generator.fq_cluster import FQCluster
from fake_query_generator.fq_param import FQParam


def proc(parsed_query: parsed_queries.ParsedQueries, reconst: ReconstCoord, new_url: str,
         trial_num: int, meth: str, use_jac: bool) -> tuple[float, float]:
    log_arr = np.zeros(trial_num)
    for i in tqdm.tqdm(range(trial_num), desc=f"jac={use_jac},meth={meth}"):
        copied_pq = copy.deepcopy(parsed_query)
        start = time.time()
        copied_pq.add_query(parser.to_dict([new_url])[0])
        new_dist = copied_pq.get_total_dist(reconst)
        try:
            _ = new_dist.reconstruct_coord(meth, use_jac)
        except ValueError:
            return float("nan"), float("nan")

        etime = time.time() - start
        log_arr[i] = etime
    return float(log_arr.mean()), float(log_arr.std())


def plot(methods: list[str], means: np.ndarray, stds: np.ndarray):
    fig, ax = plt.subplots()
    x = np.arange(len(methods)).astype(np.float32)
    kw = {"width": 0.4, "capsize": 4, "tick_label": methods, "log": True, "align": "center"}
    ax.bar(x, means[0], yerr=stds[0], label="jac: False", **kw)
    ax.bar(x + kw["width"], means[1], yerr=stds[1], label="jac: True", **kw)
    plt.xticks(x + kw["width"]/2, methods, rotation=-30)
    plt.ylabel("Calculating time [s]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fake = faker.Faker()
    fq_gen = FQGenerator(
        [
            FQCluster(
                [
                    FQParam("name1", fake.name),
                    FQParam("use-something", fake.boolean, 0.5, False),
                    FQParam("reconst", fake.coordinate, do_url_encode=False)
                ],
                0.4
            ),
            FQCluster(
                [
                    FQParam("name2", fake.name),
                    FQParam("use-something", fake.boolean, 0.5, False)
                ],
                0.6
            )
        ]
    )

    trials = 3
    url_size = 48
    method_list = ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr",
                   "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]
    urls = fq_gen.generate(url_size)
    _old_url = urls[1:]
    _new_url = urls[-1]
    parsed = parser.to_dict(_old_url)
    pq = parsed_queries.ParsedQueries(parsed)
    dist = pq.get_total_dist()
    coord = dist.reconstruct_coord()

    gla_mean = np.zeros((2, len(method_list)), dtype=np.float64)
    gla_stad = np.zeros((2, len(method_list)), dtype=np.float64)
    for ii in range(len(method_list)):
        gla_mean[0, ii], gla_stad[0, ii] = proc(pq, coord, _new_url, trials, method_list[ii], False)
        gla_mean[1, ii], gla_stad[1, ii] = proc(pq, coord, _new_url, trials, method_list[ii], True)

    plot(method_list, gla_mean, gla_stad)
