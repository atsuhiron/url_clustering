import dataclasses

import numpy as np


@dataclasses.dataclass
class GLapParam:
    lower_threshold: float
    upper_threshold: float


def _calc_similarity(dist: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    eye = np.eye(len(dist))
    similarity = 1 / (dist + eye) - eye
    similarity = similarity / np.max(similarity)
    similarity[similarity < lower_threshold] = 0.0
    similarity[similarity > upper_threshold] = 1.0
    return similarity


def calc_g_laplacian(dist: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    similarity = _calc_similarity(dist, lower_threshold, upper_threshold)
    deg_mat = np.diag(np.sum(similarity, axis=1))
    return deg_mat - similarity


def calc_sym_norm_g_laplacian(dist: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    similarity = _calc_similarity(dist, lower_threshold, upper_threshold)
    deg_mat = np.diag(np.power(np.sum(similarity, axis=1), -0.5))
    return np.eye(len(similarity)) - ((deg_mat @ similarity) @ deg_mat)
