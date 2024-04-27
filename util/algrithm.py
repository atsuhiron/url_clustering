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
