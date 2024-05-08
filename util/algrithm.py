import numpy as np
import numba


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


@numba.njit("f8[:](f8[:], f8)")
def norm_pdf(x: np.ndarray, scale: float) -> np.ndarray:
    coef = np.power(2 * np.pi, -0.5)  # scale (=sigma) は後で掛け算してキャンセルされるので最初から計算しない
    return coef * np.exp(-x * x / (2 * scale * scale))


def caldera(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> float | np.ndarray:
    eps = np.finfo(np.float64).eps
    r = np.linalg.norm(x - mean) + eps
    val = -1 * sigma * norm_pdf(r - shift, sigma)
    return val


def d_caldera(x: np.ndarray, sigma: float, mean: np.ndarray, shift: float) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    r = np.linalg.norm(x - mean)[:, np.newaxis] + eps
    val = sigma * norm_pdf(r - shift, sigma)
    d_coef = 2 * (r - shift) / sigma / sigma / r * x
    return val * d_coef


if __name__ == "__main__":
    import scipy.stats as ss


    def sigma_multiplied_norm_pdf(x, scale):
        return scale * ss.norm.pdf(x, scale=scale, loc=0)


    num = 4
    random_2d_x = np.random.random((num, 2)).astype(np.float64) - 0.5
    random_sigma = np.random.random(num).astype(np.float64)

    for ri in range(num):
        expected = sigma_multiplied_norm_pdf(random_2d_x[ri], random_sigma[ri])
        actual = norm_pdf(random_2d_x[ri], float(random_sigma[ri]))
        print(expected, actual)
