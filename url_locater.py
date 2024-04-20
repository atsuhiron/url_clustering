import numpy as np


def locate_svd(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(dist, compute_uv=True, hermitian=True)
    diag_mat = np.diag(np.sqrt(s))
    return u @ diag_mat, np.argsort(s)


def locate_eigh(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = np.linalg.eigh(dist)
    abs_eigen_val = np.abs(res.eigenvalues)
    diag_mat = np.diag(np.sqrt(abs_eigen_val))
    return (res.eigenvectors @ diag_mat).real, np.argsort(abs_eigen_val)
