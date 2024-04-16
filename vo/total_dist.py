import numpy as np

import url_locater


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

    def reconstruct_coord(self) -> np.ndarray:
        return url_locater.locate_eigh(self.dist)

    def calc_ccr(self) -> np.ndarray:
        # cumulative contribution ratio
        pass
