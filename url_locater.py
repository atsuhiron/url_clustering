import numpy as np


def locate_svd(dist: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(dist, compute_uv=True, hermitian=True)
    diag_mat = np.diag(np.sqrt(s))
    return u @ diag_mat


def locate_eigh(dist: np.ndarray) -> np.ndarray:
    res = np.linalg.eig(dist)
    diag_mat = np.diag(np.sqrt(np.abs(res.eigenvalues)))
    return (res.eigenvectors @ diag_mat).real
