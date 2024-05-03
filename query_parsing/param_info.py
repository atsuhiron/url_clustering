from __future__ import annotations
import dataclasses
import copy

from query_parsing.param_type import ParamType


@dataclasses.dataclass
class ParamInfo:
    key: str
    p_type: ParamType = ParamType.Normal
    duplication_index: int = 0

    def __hash__(self):
        return hash((self.key, self.duplication_index))

    def __eq__(self, other):
        if isinstance(other, ParamInfo):
            return self.key == other.key and self.duplication_index == other.duplication_index
        return False

    def __str__(self):
        if self.duplication_index == 0:
            return f"ParamInfo({self.key}, {self.p_type.name})"
        return f"ParamInfo({self.key}, {self.p_type.name}, dup_idx={self.duplication_index})"


class SummarisedParamInfo:
    def __init__(self, key: str, count: int, p_type: ParamType, samples: list[str], non_null_indices: set[int], duplication_index: int = 0):
        self.key = key
        self.count = count
        self.p_type = p_type
        self.duplication_index = duplication_index
        self.samples = samples
        self.non_null_indices = non_null_indices
        self.dist_arr = None

    def create_none_fill_samples(self, max_size: int) -> list[str | None]:
        nfs = [None] * max_size
        for idx, sample in zip(self.non_null_indices, self.samples):
            nfs[idx] = sample
        return nfs

    def create_param_info(self) -> ParamInfo:
        # p_type は ParamInfo の同値判定に使われないので本当はなんでもよい
        return ParamInfo(self.key, self.p_type, self.duplication_index)

    def update_by_new_query(self, new_sample: str | None):
        self.count += 1

        if new_sample is None:
            # 新しいクエリにこのパラメータがなかった場合、カウントだけインクリメントしておしまい
            return

        new_samples = copy.deepcopy(self.samples)
        new_samples.append(new_sample)
        self.samples = new_samples

        new_nni = copy.deepcopy(self.non_null_indices)
        new_nni.add(self.count)
        self.non_null_indices = new_nni

    @staticmethod
    def from_p_info(p_info: ParamInfo, count: int, p_type: ParamType, samples: list[str],
                    non_null_indices: set[int]) -> SummarisedParamInfo:
        assert len(samples) == len(non_null_indices)
        return SummarisedParamInfo(p_info.key, count, p_type, samples, non_null_indices, p_info.duplication_index)

    def __str__(self):
        if self.duplication_index == 0:
            return f"SummarisedParamInfo({self.key}, {self.p_type.name}, count={self.count})"
        return f"SummarisedParamInfo({self.key}, {self.p_type.name}, count={self.count}, dup_idx={self.duplication_index})"
