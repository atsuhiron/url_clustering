import numpy as np

from vo.location_result import BatchLocationResult


def _reverse_argsort(arr: np.ndarray) -> np.ndarray:
    return np.argsort(-arr)


def _find_first_negative_index(original_ev: np.ndarray, reversed_order: np.ndarray) -> int:
    print(original_ev)
    reverse_sorted = original_ev[reversed_order]
    for ii in range(len(original_ev)):
        if reverse_sorted[ii] < 0:
            return ii
    return len(original_ev)


def locate_svd(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(dist, compute_uv=True, hermitian=True)
    diag_mat = np.diag(np.sqrt(s))
    return u @ diag_mat, _reverse_argsort(s)


def locate_eigh(dist: np.ndarray) -> BatchLocationResult:
    res = np.linalg.eigh(dist)
    abs_eigen_val = np.abs(res.eigenvalues)
    diag_mat = np.diag(np.sqrt(abs_eigen_val))
    reversed_order = _reverse_argsort(abs_eigen_val)
    return BatchLocationResult(
        res.eigenvectors @ diag_mat,
        reversed_order,
        _find_first_negative_index(res.eigenvalues, reversed_order)
    )


if __name__ == "__main__":
    _arr = np.array([-0.5, -3.2, 7, 2])
    rev_ord = _reverse_argsort(np.abs(_arr))
    actual = _find_first_negative_index(_arr, rev_ord)
    assert actual == 1
