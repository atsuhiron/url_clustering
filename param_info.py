import dataclasses

from param_type import ParamType


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
