import numpy as np
from tqdm import tqdm

from fake_query_generator.fq_cluster import FQCluster


class FQGenerator:
    def __init__(self, clusters: list[FQCluster]):
        assert len(clusters) > 0
        self._cluster = clusters

    def generate(self, num: int) -> list[str]:
        if len(self._cluster) == 1:
            c_indices = np.zeros(num, dtype=np.int32)
        else:
            cumulative_ratio = np.cumsum([cl.query_appearance_rate for cl in self._cluster], dtype=np.float64)
            cumulative_ratio /= cumulative_ratio[-1]
            c_indices = np.zeros(num, dtype=np.int32)
            random = np.random.random(num)
            for ci in range(len(self._cluster) - 1):
                c_indices[random > cumulative_ratio[ci]] = ci + 1

        fq_list = []
        for ci in tqdm(c_indices, desc="Gen fake query"):
            fq_list.append(self._cluster[ci].generate())
        return fq_list


if __name__ == "__main__":
    import faker
    from fake_query_generator.fq_param import FQParam

    fake = faker.Faker()
    fqg = FQGenerator(
        [
            FQCluster([FQParam("name1", fake.name)], 0.4),
            FQCluster([FQParam("name2", fake.name)], 0.6),
            FQCluster([FQParam("name3", fake.name)], 0.7),
        ]
    )

    ret = fqg.generate(40)
    print(ret)
