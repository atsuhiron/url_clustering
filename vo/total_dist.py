import numpy as np

import url_locater
from vo.reconst_coord import ReconstCoord


class TotalDist:
    def __init__(self, dist: np.ndarray, old_dist: np.ndarray | None, old_coord: np.ndarray | None = None):
        self.dist = dist
        self.coord = None
        self.old_dist = old_dist
        self.old_coord = old_coord

    @property
    def ndim(self) -> int:
        return self.dist.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dist.shape

    def __len__(self) -> int:
        return len(self.dist)

    def reconstruct_coord(self) -> ReconstCoord:
        if self.old_dist is None or self.old_coord is None:
            self.old_coord, order = url_locater.locate_eigh(self.dist)
            return ReconstCoord(self.coord, order, self.dist)

        # TODO: ここに付加処理をを実装
        old_deg = len(self.old_dist)
        new_deg = len(self.dist)
        additional_dist = self.dist[old_deg:, 0: old_deg]  # shape: (n - o, o)
