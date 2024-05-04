from typing import Callable

import scipy.stats as ss
import scipy.optimize as so
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
    val = -1 * sigma * np.squeeze(ss.norm.pdf(r - shift, scale=sigma, loc=0))
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


def xzip(*ndarrays) -> np.ndarray:
    flattens = [x.flatten() for x in ndarrays]
    return np.c_[*flattens]


def _init_point_locator(old_coord: np.ndarray) -> np.ndarray:
    index = np.random.randint(0, len(old_coord) - 1)
    return old_coord[index]


def optimize_new_location(old_coord: np.ndarray, additional_dist: np.ndarray) -> np.ndarray:
    """
    `old_coord` で記述された座標点それぞれに対して、距離が `additional_dist` の座標点を求める。

    Parameters
    ----------
    old_coord : np.ndarray
        shape は `(N, M)` で N 個の座標点がそれぞれ M 次元の成分を持っている。

    additional_dist : np.ndarray
        shape は `(N,)` で N 個の座標点からの距離を表した配列。

    Returns
    -------
    new_coord : np.ndarray
        shape は `(M,)` で `additional_dist` で与えられた情報を元に新しい座標点を計算して返す。
    """
    assert old_coord.ndim == 2
    assert additional_dist.ndim == 1
    assert len(old_coord) == len(additional_dist)

    peak_coef = 1.0
    ccf = complex_caldera_enclosure(
        sigma_arr=additional_dist * peak_coef,  # この設定には議論の余地がある
        mean_arr=old_coord,
        shift_arr=additional_dist
    )

    # TODO: もうちょい凝った手法にする (ex. 最初は大きい sigma をだんだん小さくして局所解にはまらないようにする。複数の初期値から始める。)
    init_x = _init_point_locator(old_coord)
    opt_res = so.minimize(ccf, init_x)
    return opt_res.x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ccf = complex_caldera_enclosure(
        sigma_arr=np.array([0.3, 0.9, 2.2]),
        mean_arr=np.array([[-3, -3], [2.2, 1.6], [0.4, -0.2]]),
        shift_arr=np.array([1.0, 2.0, 2.5])
    )

    size = 101
    xx, yy = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))
    arr = np.zeros((size ** 2, 2))
    for ii, (_x, _y) in enumerate(zip(xx.flatten(), yy.flatten())):
        arr[ii, 0] = _x
        arr[ii, 1] = _y

    distribution = ccf(xzip(xx, yy)).reshape((size, size))

    poi = np.array([2.5, -0])
    # soret = so.least_squares(ccf, poi, verbose=2)
    soret = so.minimize(ccf, poi)
    print(soret)

    plt.pcolor(xx, yy, distribution)
    plt.plot([soret.x[0]], [soret.x[1]], "o")
    plt.show()
