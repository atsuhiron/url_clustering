from collections import Counter

from param_info import ParamInfo
from param_info import SummarisedParamInfo


class ParsedQueries:
    def __init__(self, queries: list[dict[ParamInfo, str]]):
        self._queries = queries
        self.summary = self.summarize(self._queries)

    @staticmethod
    def summarize(queries: list[dict[ParamInfo, str]]) -> list[SummarisedParamInfo]:
        # 転置辞書作成
        p_info_set = set()
        for query in queries:
            sub_param_set = set(query.keys())
            p_info_set.update(sub_param_set)
        p_info_to_p_type_map = {p_info: [] for p_info in p_info_set}
        p_info_to_samples_map = {p_info: [] for p_info in p_info_set}

        # 転置
        # p_info_to_p_type_map = {p_info: [Normal, Normal, ...], p_info: [Boolean, Boolean, ...]}
        # p_info_to_samples_map = {p_info: [234, 55, ...], p_info: [true, false, ...]}
        for param_dict in queries:
            for p_info in param_dict.keys():
                p_info_to_p_type_map[p_info].append(p_info.p_type)
                p_info_to_samples_map[p_info].append(param_dict[p_info])

        # 最頻値を集計
        p_info_to_mft_map = {}
        for p_info in p_info_to_p_type_map.keys():
            counter = Counter(p_info_to_p_type_map[p_info])
            max_count = 0
            most_freq_type = None
            for p_type in counter.keys():
                if counter[p_type] > max_count:
                    max_count = counter[p_type]
                    most_freq_type = p_type
            p_info_to_mft_map[p_info] = most_freq_type

        summary_list = []
        for p_info in p_info_set:
            summary = SummarisedParamInfo.from_p_info(
                p_info,
                len(p_info_to_samples_map[p_info]),
                p_info_to_mft_map[p_info],
                p_info_to_samples_map[p_info]
            )
            summary_list.append(summary)

        return summary_list