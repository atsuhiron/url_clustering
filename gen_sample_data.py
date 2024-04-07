import numpy as np


def gen_data(size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    サンプル用データを生成する

    Parameters
    ----------
    size: int
        生成するデータ点の数

    Returns
    -------
    coord, distance: tuple[np.ndarray, np.ndarray]
        それぞれデータ点の座標とお互いのデータ点との距離
    """
    data = np.random.random((size, 2))
    return data - np.array([data.mean(axis=0)]), _calc_dist_sq(data)


def _calc_dist_sq(pos: np.ndarray) -> np.ndarray:
    num = len(pos)
    dim = len(pos[0])

    hori = np.ones((num, num, dim), dtype=np.float32) * pos[np.newaxis, :]
    vert = np.ones((num, num, dim), dtype=np.float32) * pos[:, np.newaxis]
    relative_pos = hori - vert
    distance_sq_by_dim = relative_pos ** 2
    return distance_sq_by_dim.sum(axis=2)


if __name__ == "__main__":
    sample, dist = gen_data(6)
    print(sample)
    print(dist)
    