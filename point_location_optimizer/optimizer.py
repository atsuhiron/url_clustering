from typing import Callable

import scipy.optimize as so
import numpy as np

from util import algrithm as alg


def complex_caldera_enclosure(sigma_arr: np.ndarray, mean_arr: np.ndarray, shift_arr: np.ndarray,
                              interpret_x_as_arr: bool = False) -> tuple[Callable, Callable]:
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
    interpret_x_as_arr : bool

    Returns
    -------
    functions : tuple[Callable, Callable]
        目的関数とそのヤコビアン。
    """
    assert sigma_arr.ndim == 1
    assert mean_arr.ndim == 2
    assert shift_arr.ndim == 1
    assert len(sigma_arr) == len(mean_arr)
    assert len(sigma_arr) == len(shift_arr)

    if interpret_x_as_arr:
        def closure(x: np.ndarray) -> float:
            val = float(0)
            for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
                val += alg.caldera_non_jit(x, sigma, mean, shift)
            return val

        def d_closure(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2:
                d_vec = np.zeros((len(x), mean_arr.shape[1]))
            else:
                d_vec = np.zeros(mean_arr.shape[1])
            for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
                d_vec += alg.d_caldera_non_jit(x, sigma, mean, shift)
            return d_vec

        return closure, d_closure
    def closure(x: np.ndarray) -> float:
        return alg.sum_caldera(x, sigma_arr, mean_arr, shift_arr)

    def d_closure(x: np.ndarray) -> np.ndarray:
        return alg.sum_d_caldera(x, sigma_arr, mean_arr, shift_arr)
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

    ccf_nj, d_ccf_nj = complex_caldera_enclosure(
        sigma_arr=np.array([0.3, 0.9, 2.2]),
        mean_arr=np.array([[-3, -3], [2.2, 1.6], [0.4, -0.2]]),
        shift_arr=np.array([1.0, 2.0, 2.5]),
        interpret_x_as_arr=True
    )

    ccf, d_ccf = complex_caldera_enclosure(
        sigma_arr=np.array([0.3, 0.9, 2.2]),
        mean_arr=np.array([[-3, -3], [2.2, 1.6], [0.4, -0.2]]),
        shift_arr=np.array([1.0, 2.0, 2.5]),
        interpret_x_as_arr=False
    )

    size = 101
    xx, yy = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))
    arr = np.zeros((size ** 2, 2))
    for ii, (_x, _y) in enumerate(zip(xx.flatten(), yy.flatten())):
        arr[ii, 0] = _x
        arr[ii, 1] = _y

    distribution = ccf_nj(xzip(xx, yy)).reshape((size, size))
    _d_distribution = d_ccf_nj(xzip(xx, yy))
    d_distr_x = _d_distribution[:, 0].reshape((size, size))
    d_distr_y = _d_distribution[:, 1].reshape((size, size))

    poi = np.array([2.5, -0])
    soret = so.minimize(ccf, poi, jac=d_ccf, method="BFGS")
    print(soret)
    plt.pcolor(xx, yy, distribution)
    plt.plot([soret.x[0]], [soret.x[1]], "o")
    plt.show()

    plt.plot(xx[0], distribution[int(soret.x[1])], label="f(x)")
    plt.plot(xx[0], d_distr_x[int(soret.x[1])], label="df(x)/dx)")
    plt.legend()
    plt.show()
