import datetime
import time

import numpy as np
import faker
import tqdm
import matplotlib.pyplot as plt

from query_parsing import parsed_queries, parser
from fake_query_generator.fq_generator import FQGenerator
from fake_query_generator.fq_cluster import FQCluster
from fake_query_generator.fq_param import FQParam


def proc_remake(trial_num: int, urls: list[str]) -> tuple[float, float]:
    log_arr = np.zeros(trial_num)
    for i in tqdm.tqdm(range(trial_num), desc=f"remake: {len(urls)}"):
        start = time.time()

        parsed = parser.to_dict(urls)
        pq = parsed_queries.ParsedQueries(parsed)
        dist = pq.get_total_dist()
        _ = dist.reconstruct_coord()

        etime = time.time() - start
        log_arr[i] = etime

    return float(log_arr.mean()), float(log_arr.std())


def proc_add(trial_num: int, urls: list[str], use_jac: bool) -> tuple[float, float]:
    log_arr = np.zeros(trial_num)
    old_url = urls[1:]
    new_url = urls[:1]
    for i in tqdm.tqdm(range(trial_num), desc=f"add(jac={use_jac}): {len(urls)}"):
        parsed = parser.to_dict(old_url)
        pq = parsed_queries.ParsedQueries(parsed)
        dist = pq.get_total_dist()
        coord = dist.reconstruct_coord()

        start = time.time()
        pq.add_query(parser.to_dict(new_url)[0])
        new_dist = pq.get_total_dist(coord)
        _ = new_dist.reconstruct_coord(use_jac)

        etime = time.time() - start
        log_arr[i] = etime

    return float(log_arr.mean()), float(log_arr.std())


def plot(sizes: np.ndarray, means: np.ndarray, stds: np.ndarray):
    plt.errorbar(sizes, means[0], stds[0], fmt="o", label="Remake")
    plt.errorbar(sizes, means[1], stds[1], fmt="o", label="Add")
    plt.errorbar(sizes, means[2], stds[2], fmt="o", label="Add use_jac")

    plt.xscale("log")
    plt.xlabel("Number of data")
    plt.yscale("log")
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

    trials = 10
    sizes = np.logspace(3, 7, 17, base=4).astype(np.int64)

    gla_mean = np.zeros((3, len(sizes)), dtype=np.float64)
    gla_stad = np.zeros((3, len(sizes)), dtype=np.float64)
    for ii in range(len(sizes)):
        common_url = fq_gen.generate(int(sizes[ii]), verbose=False)

        gla_mean[0, ii], gla_stad[0, ii] = proc_remake(trials, common_url)
        gla_mean[1, ii], gla_stad[1, ii] = proc_add(trials, common_url, False)
        gla_mean[2, ii], gla_stad[2, ii] = proc_add(trials, common_url, True)

    print(datetime.datetime.now())
    plot(sizes, gla_mean, gla_stad)
