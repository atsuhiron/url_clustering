import dataclasses

import numpy as np

from fake_query_generator.fq_param import FQParam


@dataclasses.dataclass
class FQCluster:
    params: list[FQParam]
    query_appearance_rate: float

    def generate(self) -> str:
        appearance_random = np.random.random(len(self.params))

        args = []
        for i in range(len(self.params)):
            if appearance_random[i] > self.params[i].param_appearance_rate:
                continue
            args.append(self.params[i].generate())
        return "&".join(args)
