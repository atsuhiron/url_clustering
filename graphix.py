import matplotlib.pyplot as plt
import numpy as np


def draw_dist_mat(dist: np.ndarray):
    plt.imshow(dist)
    plt.show()


def draw_coord(coord: np.ndarray):
    # TODO: クラスタリングに対応させる
    assert coord.ndim == 2, f"Not supported shape: {coord.shape}"
    assert coord.shape[1] > 1, f"Not supported shape: {coord.shape}"

    plt.plot(coord[:, 0], coord[:, 1], "o")
    plt.show()


def draw_coord_for_sample(coord: np.ndarray, original: np.ndarray):
    assert coord.ndim == 2, f"Not supported shape: {coord.shape}"
    assert coord.shape[1] > 1, f"Not supported shape: {coord.shape}"
    assert len(coord) == len(original), f"Wrong length"

    for i in range(len(coord)):
        plt.plot([coord[i, 0], original[i, 0]], [coord[i, 1], original[i, 1]], ls="-", color="gray")
    plt.plot(coord[:, 0], coord[:, 1], "o", label="reconstructed")
    plt.plot(original[:, 0], original[:, 1], "o", label="original")
    plt.show()
