from typing import Callable

import scipy.stats as ss
import numpy as np


def caldera(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> float | np.ndarray:
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([[x]])
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        x = x[np.newaxis, :]
    elif isinstance(x, np.ndarray) and x.ndim != 2:
        assert False, f"Invalid ndarray shape of x: {x.shape}"

    if isinstance(mean, float) or isinstance(mean, int):
        mean = np.array([mean])
    elif not isinstance(mean, np.ndarray):
        raise TypeError
    mean = mean[np.newaxis, :]
    assert x.shape[1] == mean.shape[1], f"Invalid ndarray shape of mean: {mean.shape}"

    r = np.linalg.norm(x - mean, axis=1)
    val = np.squeeze(ss.norm.pdf(r - shift, scale=sigma, loc=0))
    if val.shape == (1,):
        return val[0]
    return val


def complex_caldera_enclosure(sigma_arr: np.ndarray, mean_arr: np.ndarray, shift_arr: np.ndarray) -> Callable:
    """
    最適化する目的関数を生成するエンクロージャ。
    N個のカルデラ関数を内包している。

    Parameters
    ----------
    sigma_arr : np.ndarray
        山の幅を表す。shape は `(N,)`。
    mean_arr : np.ndarray
        中心座標を表す。shape は `(N, M)`。
        ただし `M` はパラメータ数と同じであることが多いので、ほとんどの場合 `M=N` 。
    shift_arr : np.ndarray
        中心から山までの距離を表す。shape は `(N,)`。

    Returns
    -------
    function : Callable
        目的関数。
    """
    def closure(x: np.ndarray) -> float:
        val = float(0)
        for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
            val += caldera(x, sigma, mean, shift)
        return val
    return closure


if __name__ == "__main__":
    xx, yy = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
    arr = np.zeros((101 ** 2, 2))
    for ii, (_x, _y) in enumerate(zip(xx.flatten(), yy.flatten())):
        arr[ii, 0] = _x
        arr[ii, 1] = _y

    ret = caldera(arr, 1.1, np.array([0, 0]), 2.0)