from collections import Counter

import numpy as np
import rapidfuzz.process as rapidfuzz_p
import rapidfuzz.distance as rapidfuzz_d

from query_parsing.param_type import ParamType
from query_parsing.param_info import ParamInfo
from query_parsing.param_info import SummarisedParamInfo
from vo.total_dist import TotalDist


class ParsedQueries:
    def __init__(self, queries: list[dict[ParamInfo, str]]):
        self._queries = queries
        self.summary, self.p_info_set = self.create_summary(self._queries, len(self._queries))
        self.total_dist = self.calc_total_dist(self.summary, len(self._queries))

    def __len__(self) -> int:
        return len(self.summary)

    def get_total_dist(self, total_dist: TotalDist | None = None) -> TotalDist:
        return TotalDist(self.total_dist, total_dist.dist, total_dist.old_coord)

    def add_query(self, query: dict[ParamInfo, str], strict_mode: bool = False):
        # 新規パラメータが無いか確認
        new_params = []
        for new_p_info in query.keys():
            if new_p_info not in self.p_info_set:
                new_params.append(new_p_info)
        if len(new_params) > 0:
            msg = "This query contain new parameter: " + str([n_param.key for n_param in new_params])
            if strict_mode:
                raise ValueError(msg)
            else:
                print("Warning: " + msg)

        # 新しいクエリを既存の SummarisedParamInfo に組み込む
        for sp_info in self.summary:
            # 既存の SummarisedParamInfo の更新
            _p_info = sp_info.create_param_info()
            param_value: str | None = query.get(_p_info)
            sp_info.update_by_new_query(param_value)

            # dist の更新
            none_fill_sample = sp_info.create_none_fill_samples(len(self._queries))
            dist_arr_1d = ParsedQueries._calc_paramwise_dist_core(sp_info.p_type, none_fill_sample, [param_value])
            sp_info.dist_arr = self.attach_arr_to_mat(sp_info.dist_arr, np.squeeze(dist_arr_1d))

        self.total_dist = self.calc_total_dist(self.summary, len(self._queries))

    @staticmethod
    def create_summary(queries: list[dict[ParamInfo, str]], query_size: int) -> tuple[list[SummarisedParamInfo], set[ParamInfo]]:
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

        return ParsedQueries._calc_paramwise_dist(summary_list, query_size), p_info_set

    @staticmethod
    def _calc_paramwise_dist_core(param_type: ParamType, sample_ref: list[str | None], sample_tgt: list[str | None] | None = None) -> np.ndarray:
        """

        Parameters
        ----------
        param_type : ParamType
            パラメータの種類。現状では使ってない。
        sample_ref : list[str | None]
            参照する方のパラメータ。全てのクエリサンプルについて計算する必要があるので、該当するパラメータを使用していないクエリは `None` で埋める必要がある。
        sample_tgt : list[str | None] | None
            確認する方のパラメータ。一括生成の時は使用しない。デフォルト値は `None`。

        Returns
        -------
        dist : np.ndarray
            shape は `(len(sample_tgt), len(sample_ref))`。
        """
        # TODO: ParamType ごとの変換処理を入れる
        if sample_tgt is None:
            sample_tgt = sample_ref
        return rapidfuzz_p.cdist(sample_tgt, sample_ref, scorer=rapidfuzz_d.JaroWinkler.normalized_distance)

    @staticmethod
    def _calc_paramwise_dist(summary: list[SummarisedParamInfo], query_size: int) -> list[SummarisedParamInfo]:
        for sp_info in summary:
            none_fill_sample = sp_info.create_none_fill_samples(query_size)
            dist_arr = ParsedQueries._calc_paramwise_dist_core(sp_info.p_type, none_fill_sample)
            np.fill_diagonal(dist_arr, 0)
            sp_info.dist_arr = dist_arr
        return summary

    @staticmethod
    def calc_total_dist(summary: list[SummarisedParamInfo], query_size: int) -> np.ndarray:
        total_dist = np.zeros((len(summary), query_size, query_size))
        for pi, param in enumerate(summary):
            total_dist[pi] = param.dist_arr
        return total_dist.sum(axis=0)

    @staticmethod
    def attach_arr_to_mat(mat: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """
        `(N, N)` の対称行列と `(N,)` の配列から `(N+1, N+1)` の対称行列を作る。

        [[ m11, m12, m13 ],
         [ m12, m22, m23 ],
         [ m13, m23, m33 ]]
        と
        [ a1, a2, a3 ]
        から
        [[ m11, m12, m13, a1 ],
         [ m12, m22, m23, a2 ],
         [ m13, m23, m33, a3 ]
         [ a1,  a2,  a3,  0  ]]
        を作る。

        Parameters
        ----------
        mat : np.ndarray
            `(N, N)` の対称行列。
        arr : np.ndarray
            `(N,)` の1次元配列。

        Returns
        -------
        attached_mat : np.ndarray
            `(N+1, N+1)` の対称行列

        """
        assert mat.ndim == 2
        assert mat.shape[0] == mat.shape[1]
        assert arr.ndim == 1
        assert len(arr) == len(mat)

        n = len(arr)
        attached_mat = np.zeros((n + 1, n + 1), dtype=mat.dtype)
        attached_mat[0: n, 0: n] = mat
        attached_mat[0: n, n] = arr
        attached_mat[n, 0: n] = arr
        return attached_mat
