import numpy as np
import scipy.stats as ss
import numba


EPS = np.finfo(np.float64).eps


@numba.njit("f8[:,:](f8[:,:])")
def calc_dist_sq(pos: np.ndarray) -> np.ndarray:
    num = len(pos)
    dim = len(pos[0])

    hori = np.ones((num, num, dim), dtype=np.float64) * pos[np.newaxis, :]
    vert = np.ones((num, num, dim), dtype=np.float64) * pos[:, np.newaxis]
    relative_pos = hori - vert
    distance_sq_by_dim = relative_pos ** 2
    return distance_sq_by_dim.sum(axis=2)


@numba.jit("f8[:](f8[:,:],f8[:,:])", nopython=True, parallel=True)
def calc_reconstruction_error_core(coord: np.ndarray, dist: np.ndarray) -> np.ndarray:
    errors = np.zeros(len(coord), dtype=np.float64)
    for ii in numba.prange(len(coord)):
        rec_dist = calc_dist_sq(coord[:, 0: ii + 1])
        errors[ii] = np.sum(np.square(dist - rec_dist))
    return np.sqrt(errors)


@numba.njit("f8(f8, f8)")
def norm_pdf(x: float, scale: float) -> float:
    coef = np.power(2 * np.pi, -0.5)  # scale (=sigma) は後で掛け算してキャンセルされるので最初から計算しない
    return coef * np.exp(-x * x / (2 * scale * scale))


@numba.njit("f8[:](f8[:], f8)")
def norm_pdf_arr(x: np.ndarray, scale: float) -> np.ndarray:
    coef = np.power(2 * np.pi, -0.5)  # scale (=sigma) は後で掛け算してキャンセルされるので最初から計算しない
    return coef * np.exp(-x * x / (2 * scale * scale))


@numba.njit("f8(f8[:], f8, f8[:], f8)")
def caldera(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> float:
    eps = np.finfo(np.float64).eps
    r = np.linalg.norm(x - mean) + eps
    val = -1 * sigma * norm_pdf(r - shift, sigma)
    return val


@numba.njit("f8(f8[:], f8[:], f8[:,:], f8[:])")
def sum_caldera(x: np.ndarray, sigma_arr: np.ndarray, mean_arr: np.ndarray, shift_arr: np.ndarray) -> float:
    val = float(0)
    for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
        val += caldera(x, sigma, mean, shift)
    return val


@numba.njit("f8[:](f8[:], f8, f8[:], f8)")
def d_caldera(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> float:
    eps = np.finfo(np.float64).eps
    r = np.linalg.norm(x - mean) + eps
    val = sigma * norm_pdf(r - shift, sigma)
    d_coef = 2 * (r - shift) / sigma / sigma / r * x
    return val * d_coef


@numba.njit("f8[:](f8[:], f8[:], f8[:,:], f8[:])")
def sum_d_caldera(x: np.ndarray, sigma_arr: np.ndarray, mean_arr: np.ndarray, shift_arr: np.ndarray) -> np.ndarray:
    d_vec = np.zeros(mean_arr.shape[1], dtype=np.float64)
    for sigma, mean, shift in zip(sigma_arr, mean_arr, shift_arr):
        d_vec += d_caldera(x, sigma, mean, shift)
    return d_vec


def caldera_non_jit(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> float | np.ndarray:
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


def d_caldera_non_jit(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> np.ndarray:
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
    d_coef = 2 * (r - shift) / sigma / sigma / r * x
    return np.squeeze(val * d_coef)


if __name__ == "__main__":
    def sigma_multiplied_norm_pdf(x, scale):
        return scale * ss.norm.pdf(x, scale=scale, loc=0)


    _num = 4
    random_2d_x = np.random.random((4, 2)).astype(np.float64) - 0.5
    random_sigma = np.random.random(_num).astype(np.float64)

    for ri in range(_num):
        expected = sigma_multiplied_norm_pdf(random_2d_x[ri], random_sigma[ri])
        actual = norm_pdf_arr(random_2d_x[ri], float(random_sigma[ri]))
        print(expected, actual)

        ret_c = caldera(random_2d_x[ri], float(random_sigma[ri]), np.array([0.0, 0.0]), 0.525)
        ret_dc = d_caldera(random_2d_x[ri], float(random_sigma[ri]), np.array([0.0, 0.0]), 0.525)

    sigmas = np.array([0.3, 0.9, 2.2])
    means = np.array([[-3, -3], [2.2, 1.6], [0.4, -0.2]])
    shifts = np.array([1.0, 2.0, 2.5])

    print(sum_caldera(random_2d_x[0], sigmas, means, shifts))
    print(sum_d_caldera(random_2d_x[0], sigmas, means, shifts))
