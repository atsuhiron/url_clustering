from query_parsing.param_type import ParamType
from query_parsing.param_info import ParamInfo


def estimate_p_type(value: str | None) -> ParamType:
    if value is None:
        return ParamType.Normal
    if value.lower() == "true" or value.lower() == "false":
        return ParamType.Boolean
    if value.count(".") > 1:
        return ParamType.List
    return ParamType.Normal


def _to_dict_core(query: str) -> dict[ParamInfo, str]:
    kv_pairs = {}
    key_set = set()
    for kv_str in query.split("&"):
        kv_split = kv_str.split("=")
        if len(kv_split) == 1 or (len(kv_split) == 2 and len(kv_split[1]) == 0):
            kv_pair = (kv_split[0], None)
        elif len(kv_split) == 2:
            kv_pair = (kv_split[0], kv_split[1])
        else:
            kv_pair = (kv_split[0], "=".join(kv_split[1:]))

        if len(kv_pair[0]) == 0:
            # key が空文字の場合、どうにもならないので無視
            continue

        if kv_pair[0] in key_set:
            # key が重複している場合、duplication index で識別する
            max_dup_index = max(
                map(lambda _p_info: _p_info.duplication_index, filter(lambda _p_info: _p_info.key == kv_pair[0], list(kv_pairs.keys()))))
            p_info = ParamInfo(kv_pair[0], estimate_p_type(kv_pair[1]), max_dup_index + 1)
        else:
            p_info = ParamInfo(kv_pair[0], estimate_p_type(kv_pair[1]))

        key_set.update(kv_pair[0])
        kv_pairs[p_info] = kv_pair[1]
    return kv_pairs


def to_dict(queries: list[str]) -> list[dict[ParamInfo, str]]:
    return [_to_dict_core(query) for query in queries]
