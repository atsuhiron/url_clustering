import numpy as np

import util.algrithm as alg


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
    return data - np.array([data.mean(axis=0)]), alg.calc_dist_sq(data)


if __name__ == "__main__":
    sample, dist = gen_data(6)
    print(sample)
    print(dist)
    