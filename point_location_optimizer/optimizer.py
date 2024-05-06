from typing import Callable

import scipy.stats as ss
import scipy.optimize as so
import numpy as np


EPS = np.finfo(np.float64).eps


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

    r = np.linalg.norm(x - mean, axis=1) + EPS
    val = -1 * sigma * np.squeeze(ss.norm.pdf(r - shift, scale=sigma, loc=0))
    if val.shape == (1,):
        return val[0]
    return val


def d_caldera(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> np.ndarray:
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
    r = np.linalg.norm(x - mean, axis=1)[:, np.newaxis] + EPS
    val = sigma * ss.norm.pdf(r - shift, scale=sigma, loc=0)
    d_coef = -2 * (r - shift) / sigma / sigma / r * x
    return np.squeeze(val * d_coef)


def complex_caldera_enclosure(sigma_arr: np.ndarray, mean_arr: np.ndarray, shift_arr: np.ndarray) -> tuple[Callable, Callable]:
    """
    最適化する目的関数を生成するエンクロージャ。
    N個のカルデラ関数を内包している。

    f_i(\vec{x}) = \frac{1}{\sqrt{2\pi}} \exp \right[ -\frac{(|\vec{x} - \vec{m_i}| - d)^2}{\sigma_i^2} \left]
    \frac{\partial f_i(\vec{x})}{\partial x_j} = -\frac{2(|\vec{x} - \vec{m_i}| - d)x_j}{\sigma_i^2|\vec{x} - \vec{m}|} f(\vec{x})

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
    functions : tuple[Callable, Callable]
        目的関数とそのヤコビアン。
    """
    def closure(x: np.ndarray) -> float:
        val = float(0)
        for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
            val += caldera(x, sigma, mean, shift)
        return val

    def d_closure(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            d_vec = np.zeros((len(x), mean_arr.shape[1]))
        else:
            d_vec = np.zeros(mean_arr.shape[1])
        for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
            d_vec += d_caldera(x, sigma, mean, shift)
        return d_vec
    return closure, d_closure


def xzip(*ndarrays) -> np.ndarray:
    flattens = [x.flatten() for x in ndarrays]
    return np.c_[*flattens]


def _init_point_locator(old_coord: np.ndarray) -> np.ndarray:
    index = np.random.randint(0, len(old_coord) - 1)
    return old_coord[index]


def optimize_new_location(old_coord: np.ndarray, additional_dist: np.ndarray, use_jac: bool, method: str | None = None) -> np.ndarray:
    """
    `old_coord` で記述された座標点それぞれに対して、距離が `additional_dist` の座標点を求める。

    Parameters
    ----------
    old_coord : np.ndarray
        shape は `(N, M)` で N 個の座標点がそれぞれ M 次元の成分を持っている。

    additional_dist : np.ndarray
        shape は `(N,)` で N 個の座標点からの距離を表した配列。

    use_jac : bool
        最適化時にヤコビアンを使うかどうか。

    method : str
        最適化メソッド。デフォルトでは `use_jac` が True の場合は SLSQP を利用する。
        False の場合は L-BFGS-B を使用する。

    Returns
    -------
    new_coord : np.ndarray
        shape は `(M,)` で `additional_dist` で与えられた情報を元に新しい座標点を計算して返す。
    """
    assert old_coord.ndim == 2
    assert additional_dist.ndim == 1
    assert len(old_coord) == len(additional_dist)

    peak_coef = 1.0
    ccf, d_ccf = complex_caldera_enclosure(
        sigma_arr=additional_dist * peak_coef,  # この設定には議論の余地がある
        mean_arr=old_coord,
        shift_arr=additional_dist
    )

    # TODO: もうちょい凝った手法にする (ex. 最初は大きい sigma をだんだん小さくして局所解にはまらないようにする。複数の初期値から始める。)
    init_x = _init_point_locator(old_coord)
    if use_jac:
        if method is None:
            method = "SLSQP"
        opt_res = so.minimize(ccf, init_x, jac=d_ccf, method=method)
    else:
        if method is None:
            method = "L-BFGS-B"
        opt_res = so.minimize(ccf, init_x, method=method)
    return opt_res.x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ccf, d_ccf = complex_caldera_enclosure(
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
    soret = so.minimize(ccf, poi, jac=d_ccf, method="BFGS")
    print(soret)
    plt.pcolor(xx, yy, distribution)
    plt.plot([soret.x[0]], [soret.x[1]], "o")
    plt.show()
