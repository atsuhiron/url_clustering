import dataclasses

import numpy as np


@dataclasses.dataclass
class BatchLocationResult:
    coord: np.ndarray
    order: np.ndarray
    first_negative_index: int

    def __post_init__(self):
        if self.coord.ndim != 2:
            raise ValueError
        if self.coord.shape[0] != self.coord.shape[1]:
            raise ValueError


@dataclasses.dataclass
class AppendLocationResult:
    pass
