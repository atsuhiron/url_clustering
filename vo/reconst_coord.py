import numpy as np

import util.algrithm as alg


class ReconstCoord:
    def __init__(self, coord: np.ndarray, order: np.ndarray, dist: np.ndarray):
        assert coord.ndim == 2
        assert coord.shape[0] == coord.shape[1]
        self.coord = coord
        self._dist = dist
        self._order = order

    @property
    def ndim(self) -> int:
        return self.coord.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coord.shape

    def __len__(self) -> int:
        return len(self.coord)

    def get_sorted_coord(self, deg: int = 2) -> np.ndarray:
        assert deg > 0
        _sorted = self.coord[:, self._order]
        return _sorted[:, 0:deg]

    def calc_reconstruction_error(self) -> np.ndarray:
        return alg.calc_reconstruction_error_core(self.coord, self._dist)
