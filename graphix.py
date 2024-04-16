import matplotlib.pyplot as plt
import numpy as np

from vo.total_dist import TotalDist
from vo.reconst_coord import ReconstCoord


def draw_dist_mat(dist: TotalDist):
    plt.imshow(dist.dist)
    plt.show()


def draw_coord(coord: ReconstCoord):
    # TODO: クラスタリングに対応させる
    assert coord.ndim == 2, f"Not supported shape: {coord.shape}"
    assert coord.shape[1] > 1, f"Not supported shape: {coord.shape}"

    _coord = coord.coord
    plt.plot(_coord[:, 0], _coord[:, 1], "o")
    plt.show()


def draw_reconstruction_error(reconstruction_error: np.ndarray):
    plt.plot(reconstruction_error)
    plt.yscale("log")
    plt.show()


def draw_coord_for_sample(coord_1: np.ndarray, coord_2: np.ndarray, original: np.ndarray):
    assert coord_1.ndim == 2, f"Not supported shape: {coord_1.shape}"
    assert coord_1.shape[1] > 1, f"Not supported shape: {coord_1.shape}"
    assert len(coord_1) == len(original), f"Wrong length"
    assert coord_2.ndim == 2, f"Not supported shape: {coord_2.shape}"
    assert coord_2.shape[1] > 1, f"Not supported shape: {coord_2.shape}"
    assert len(coord_2) == len(original), f"Wrong length"

    for i in range(len(coord_1)):
        plt.plot([coord_1[i, 0], coord_2[i, 0], original[i, 0]], [coord_1[i, 1], coord_2[i, 1], original[i, 1]], ls="-", color="gray")
    plt.plot(coord_1[:, 0], coord_1[:, 1], "o", label="reconstructed_1")
    plt.plot(coord_2[:, 0], coord_2[:, 1], "o", label="reconstructed_2")
    plt.plot(original[:, 0], original[:, 1], "o", label="original")
    plt.legend()
    plt.show()
