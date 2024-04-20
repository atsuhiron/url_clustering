import numpy as np

import url_locater
from vo.reconst_coord import ReconstCoord


class TotalDist:
    def __init__(self, dist: np.ndarray):
        self.dist = dist

    @property
    def ndim(self) -> int:
        return self.dist.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dist.shape

    def __len__(self) -> int:
        return len(self.dist)

    def reconstruct_coord(self) -> ReconstCoord:
        coord, order = url_locater.locate_eigh(self.dist)
        return ReconstCoord(coord, order, self.dist)
