import numpy as np

import util.algrithm as alg
from vo.location_result import BatchLocationResult


class ReconstCoord:
    def __init__(self, loc_result: BatchLocationResult, dist: np.ndarray):
        self.loc_result = loc_result
        self._dist = dist

    @property
    def shape(self) -> tuple[int, ...]:
        return self.loc_result.coord.shape

    def __len__(self) -> int:
        return len(self.loc_result.coord)

    def get_sorted_coord(self, deg: int = 2) -> np.ndarray:
        if deg < 0:
            raise ValueError(f"Degree must be positive: {deg}")
        if deg >= self.loc_result.first_negative_index:
            raise ValueError(f"Degree must be less than the first negative eigenvalue index: {deg} >= {self.loc_result.first_negative_index}")
        _sorted = self.loc_result.coord[:, self.loc_result.order]
        return _sorted[:, 0:deg]

    def calc_reconstruction_error(self) -> np.ndarray:
        return alg.calc_reconstruction_error_core(self.loc_result.coord, self._dist, self.loc_result.first_negative_index - 1)
