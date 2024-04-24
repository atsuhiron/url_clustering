import numpy as np


def _reverse_argsort(arr: np.ndarray) -> np.ndarray:
    return len(arr) - 1 - np.argsort(arr)


def locate_svd(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(dist, compute_uv=True, hermitian=True)
    diag_mat = np.diag(np.sqrt(s))
    return u @ diag_mat, _reverse_argsort(s)


def locate_eigh(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = np.linalg.eigh(dist)
    abs_eigen_val = np.abs(res.eigenvalues)
    diag_mat = np.diag(np.sqrt(abs_eigen_val))
    return (res.eigenvectors @ diag_mat).real, _reverse_argsort(abs_eigen_val)
