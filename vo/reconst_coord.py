import numpy as np

import util.algrithm as alg


class ReconstCoord:
    def __init__(self, coord: np.ndarray, dist: np.ndarray):
        assert coord.ndim == 2
        assert coord.shape[0] == coord.shape[1]
        self.coord = coord
        self._dist = dist

    @property
    def ndim(self) -> int:
        return self.coord.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coord.shape

    def __len__(self) -> int:
        return len(self.coord)

    def calc_reconstruction_error(self) -> np.ndarray:
        errors = np.zeros(len(self))
        for ii in range(len(self)):
            rec_dist = alg.calc_dist_sq(self.coord[0: ii + 1])
            errors[ii] = np.sum(np.square(self._dist - rec_dist))
        return np.sqrt(errors)
