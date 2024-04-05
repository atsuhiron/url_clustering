import numpy as np


def gen_data(size: int) -> np.ndarray:
    data = np.random.random((size, 2))
    return data - np.array([data.mean(axis=0)])


def calc_dist_sq(pos: np.ndarray) -> np.ndarray:
    num = len(pos)
    dim = len(pos[0])

    hori = np.ones((num, num, dim), dtype=np.float32) * pos[np.newaxis, :]
    vert = np.ones((num, num, dim), dtype=np.float32) * pos[:, np.newaxis]
    relative_pos = hori - vert
    distance_sq_by_dim = relative_pos ** 2
    return distance_sq_by_dim.sum(axis=2)


if __name__ == "__main__":
    sample = gen_data(6)
    dist = calc_dist_sq(sample)
    print(sample)
    print(dist)
    