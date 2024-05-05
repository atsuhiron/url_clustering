import numpy as np

import url_locater
from vo.reconst_coord import ReconstCoord
import point_location_optimizer.optimizer as opt


class TotalDist:
    def __init__(self, dist: np.ndarray, old_reconst: ReconstCoord | None):
        self.dist = dist
        self.coord = None

        if old_reconst is None:
            self.old_dist = None
            self.old_coord = None
            self.old_order = None
        else:
            self.old_dist = old_reconst.dist
            self.old_coord = old_reconst.coord
            self.old_order = old_reconst.order

    @property
    def ndim(self) -> int:
        return self.dist.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dist.shape

    def __len__(self) -> int:
        return len(self.dist)

    def is_square_coord(self) -> bool:
        if self.coord is None:
            return False
        return self.coord.shape[0] == self.coord.shape[1]

    def reconstruct_coord(self, method: str | None = None, use_jac: bool = True) -> ReconstCoord:
        if self.old_dist is None or self.old_coord is None:
            self.coord, order = url_locater.locate_eigh(self.dist)
            return ReconstCoord(self.coord, order, self.dist)

        old_deg = len(self.old_dist)
        additional_deg = len(self.dist) - old_deg
        additional_dist = self.dist[old_deg:, :old_deg]  # shape: (deg_a, deg_o)
        additional_coord = np.zeros_like(additional_dist)  # shape: (deg_a, deg_o)

        # 追加分の座標を推定
        for ni in range(additional_deg):
            additional_coord[ni] = opt.optimize_new_location(self.old_coord, additional_dist[ni], use_jac, method)

        # coord の更新
        self.coord = np.zeros((old_deg + additional_deg, old_deg))
        self.coord[:old_deg] = self.old_coord
        self.coord[old_deg:] = additional_coord

        return ReconstCoord(self.coord, self.old_order, self.dist)
