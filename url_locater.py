import numpy as np


def locate(dist: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(dist, compute_uv=True, hermitian=True)
    diag_mat = np.diag(np.sqrt(s))
    return u @ diag_mat
