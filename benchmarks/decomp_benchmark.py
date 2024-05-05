import dataclasses
import itertools
import time
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


@dataclasses.dataclass
class BenchRes:
    mode: str
    sizes: list[int]
    sec_mean: list[float]
    sec_std: list[float]
    dtype: type

    def get_alpha(self) -> float:
        if self.dtype is np.float64:
            return 0.5
        return 1.0

    def get_name(self) -> str:
        type_str = re.search(r"float\d\d", str(self.dtype))
        return f"{self.mode} {type_str.group()}"


def gen_random_symmetric_arr(size: int, dtype: type) -> np.ndarray:
    ra = np.random.random((size, size))
    ra += np.transpose(ra)
    np.fill_diagonal(ra, 0)
    return ra.astype(dtype)


def measure(arr: np.ndarray, mode: str, trial_num: int) -> tuple[float, float]:
    log_arr = np.zeros(trial_num)

    if mode == "eig":
        for i in range(trial_num):
            s = time.time()
            _ = np.linalg.eig(arr)
            log_arr[i] = time.time() - s
    elif mode == "eigh":
        for i in range(trial_num):
            s = time.time()
            _ = np.linalg.eigh(arr)
            log_arr[i] = time.time() - s
    elif mode == "svd":
        for i in range(trial_num):
            s = time.time()
            _ = np.linalg.svd(arr)
            log_arr[i] = time.time() - s
    elif mode == "svdh":
        for i in range(trial_num):
            s = time.time()
            _ = np.linalg.svd(arr, hermitian=True)
            log_arr[i] = time.time() - s
    else:
        assert False, f"Not supported mode: {mode}"

    return float(log_arr.mean()), float(log_arr.std())


def plot(brs: list[BenchRes]):
    plotted_modes: dict[str, int] = {"dummy": -1}
    for br in brs:
        if br.mode not in plotted_modes.keys():
            plotted_modes[br.mode] = 1 + max(plotted_modes.values())
        col_index = plotted_modes[br.mode]
        plt.errorbar(br.sizes, br.sec_mean, br.sec_std,
                     marker="o", ls="", capsize=2, alpha=br.get_alpha(),
                     color=matplotlib.colormaps.get_cmap("tab10")(col_index), label=br.get_name())
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix size")
    plt.ylabel("Calculating time [s]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    trial = 20
    sizes = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    modes = ["eig", "eigh", "svd", "svdh"]
    types = [np.float32, np.float64]
    br_list = []
    for _mode, _type in itertools.product(modes, types):
        print(f"Measuring: {_mode} {_type}")
        mean_list = []
        std_list = []
        for _size in sizes:
            _arr = gen_random_symmetric_arr(_size, _type)
            mean, std = measure(_arr, _mode, trial)
            mean_list.append(mean)
            std_list.append(std)

        br_list.append(BenchRes(_mode, sizes, mean_list, std_list, _type))

    plot(br_list)
    