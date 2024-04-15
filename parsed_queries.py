from collections import Counter

import numpy as np
import rapidfuzz.process as rapidfuzz_p
import rapidfuzz.distance as rapidfuzz_d

from param_info import ParamInfo
from param_info import SummarisedParamInfo
from vo.total_dist import TotalDist


class ParsedQueries:
    def __init__(self, queries: list[dict[ParamInfo, str]]):
        self._queries = queries
        self.summary = self.summarize(self._queries)
        self.summary = self.calc_paramwise_dist(self.summary, len(self._queries))
        self.total_dist = self.calc_total_dist(self.summary, len(self._queries))

    def __len__(self) -> int:
        return len(self.summary)

    def get_total_dist(self) -> TotalDist:
        return TotalDist(self.total_dist)

    @staticmethod
    def summarize(queries: list[dict[ParamInfo, str]]) -> list[SummarisedParamInfo]:
        # 転置辞書作成
        p_info_set = set()
        for query in queries:
            sub_param_set = set(query.keys())
            p_info_set.update(sub_param_set)
        p_info_to_p_type_map = {p_info: [] for p_info in p_info_set}
        p_info_to_samples_map = {p_info: [] for p_info in p_info_set}
        p_info_to_nni_map = {p_info: set() for p_info in p_info_set}

        # 転置
        # p_info_to_p_type_map = {p_info: [Normal, Normal, ...], p_info: [Boolean, Boolean, ...]}
        # p_info_to_samples_map = {p_info: [234, 55, ...], p_info: [true, false, ...]}
        for idx, param_dict in enumerate(queries):
            for p_info in param_dict.keys():
                p_info_to_p_type_map[p_info].append(p_info.p_type)
                p_info_to_samples_map[p_info].append(param_dict[p_info])
                p_info_to_nni_map[p_info].add(idx)

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
                p_info_to_samples_map[p_info],
                p_info_to_nni_map[p_info]
            )
            summary_list.append(summary)

        return summary_list

    @staticmethod
    def calc_paramwise_dist(summary: list[SummarisedParamInfo], query_size: int) -> list[SummarisedParamInfo]:
        for param in summary:
            none_fill_sample = param.create_none_fill_samples(query_size)
            # TODO: ParamType ごとの変換処理を入れる
            dist_arr = rapidfuzz_p.cdist(none_fill_sample, none_fill_sample, scorer=rapidfuzz_d.JaroWinkler.normalized_distance)
            np.fill_diagonal(dist_arr, 0)
            param.dist_arr = dist_arr
        return summary

    @staticmethod
    def calc_total_dist(summary: list[SummarisedParamInfo], query_size: int) -> np.ndarray:
        total_dist = np.zeros((len(summary), query_size, query_size))
        for pi, param in enumerate(summary):
            total_dist[pi] = param.dist_arr
        return total_dist.sum(axis=0)
