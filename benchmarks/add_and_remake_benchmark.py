from typing import Type
import abc
import datetime
import time

import numpy as np
import scipy.optimize as so
import faker
import tqdm
import matplotlib
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
        _ = new_dist.reconstruct_coord(None, use_jac)

        etime = time.time() - start
        log_arr[i] = etime

    return float(log_arr.mean()), float(log_arr.std())


class ApxFuncBase(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def f(self, *args):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_param_info() -> tuple[str, ...]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_p0() -> list[float]:
        pass


class ApxFuncPowerOffset(ApxFuncBase):
    @staticmethod
    def f(*args):
        x, a, b, c = args
        return a * np.power(x, b) + c

    @staticmethod
    def get_param_info() -> tuple[str, ...]:
        return "a", "b", "c"

    @staticmethod
    def get_p0() -> list[float]:
        return [0.1, 1.9, 0.0]


class ApxFuncPower(ApxFuncBase):
    @staticmethod
    def f(*args):
        x, a, b = args
        return a * np.power(x, b)

    @staticmethod
    def get_param_info() -> tuple[str, ...]:
        return "a", "b"

    @staticmethod
    def get_p0() -> list[float]:
        return [0.1, 1.9]


def _float_str(val: float) -> str:
    if np.abs(val) < 0.0001:
        return f"{val:.3e}"
    return f"{val:.4f}"


def plot(sizes: np.ndarray, means: np.ndarray, stds: np.ndarray, apx: Type[ApxFuncBase]):
    labels = ["Remake", "Add", "Add use_jac"]
    for pi in range(len(labels)):
        is_not_nan = np.logical_not(np.isnan(means[pi]))
        x = sizes[is_not_nan]
        y = means[pi, is_not_nan]
        opt_res = so.curve_fit(apx.f, x, y, apx.get_p0())
        apx_x = np.logspace(np.log2(sizes[0]), np.log2(sizes[-1]), 128, base=2)
        apx_y = apx.f(apx_x, *opt_res[0])
        apx_lab = ""
        for p_name, p_val in zip(apx.get_param_info(), opt_res[0]):
            apx_lab += f"{p_name}={_float_str(float(p_val))}, "
        apx_lab += f"{labels[pi]:12s}"
        plt.errorbar(sizes, means[pi], stds[pi], fmt="o", capsize=2, label=labels[pi])
        plt.plot(apx_x, apx_y, color=matplotlib.colormaps.get_cmap("tab10")(pi), label=apx_lab)

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

    trials = 3
    sizes = np.logspace(3, 7, 17, base=4).astype(np.int64)

    gla_mean = np.zeros((3, len(sizes)), dtype=np.float64) + np.nan
    gla_stad = np.zeros((3, len(sizes)), dtype=np.float64) + np.nan
    for ii in range(len(sizes)):
        common_url = fq_gen.generate(int(sizes[ii]), verbose=False)

        gla_mean[0, ii], gla_stad[0, ii] = proc_remake(trials, common_url)
        gla_mean[1, ii], gla_stad[1, ii] = proc_add(trials, common_url, False)
        gla_mean[2, ii], gla_stad[2, ii] = proc_add(trials, common_url, True)
        print("")

    np.save("benchmarks/temp_mean.npy", gla_mean)
    np.save("benchmarks/temp_stad.npy", gla_stad)
    print(datetime.datetime.now())
    try:
        plot(sizes, gla_mean, gla_stad, ApxFuncPowerOffset)
    except RuntimeError:
        plt.clf()
        plot(sizes, gla_mean, gla_stad, ApxFuncPower)
