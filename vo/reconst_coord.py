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

    def append(self, dist_from: np.ndarray):
        assert dist_from.ndim == 1
        assert dist_from.shape[0] == self.coord.shape[1]


if __name__ == "__main__":
    import scipy.stats as ss
    import numpy as np

    def caldera(x, sigma, mean, shift) -> float:
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([[x]])
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            x = x[np.newaxis, :]
        elif isinstance(x, np.ndarray) and x.ndim != 2:
            assert False, f"Invalid ndarray shape of x: {x.shape}"

        if isinstance(mean, float) or isinstance(mean, int):
            mean = np.array([mean])
        elif not isinstance(mean, np.ndarray):
            raise TypeError
        mean = mean[np.newaxis, :]
        assert x.shape[1] == mean.shape[1], f"Invalid ndarray shape of mean: {mean.shape}"

        r = np.linalg.norm(x - mean, axis=1)
        return ss.norm.pdf(r - shift, scale=sigma, loc=mean)



