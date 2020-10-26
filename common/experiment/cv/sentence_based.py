import collections
from os import path

import numpy as np

from arekit.common.experiment.cv.base import BaseCVFolding
from arekit.common.experiment.cv.doc_stat.base import BaseDocumentStatGenerator


class StatBasedCVFolding(BaseCVFolding):
    """ Sentence-based separation.
        Considering a separation in foldings,
        equal each other in terms of sentence count.
    """

    def __init__(self, docs_stat, docs_stat_filepath):
        assert(isinstance(docs_stat, BaseDocumentStatGenerator))
        assert(isinstance(docs_stat_filepath, unicode))
        super(StatBasedCVFolding, self).__init__()

        self.__docs_stat = docs_stat
        self.__docs_stat_filepath = docs_stat_filepath

    # region private methods

    @staticmethod
    def __select_group(cv_group_size, item):
        deltas = []
        for i in range(len(cv_group_size)):
            delta = StatBasedCVFolding.__calc_cv_group_delta(
                cv_group_size=cv_group_size,
                item=item,
                g_index_to_add=i)
            deltas.append(delta)

        return int(np.argmin(deltas))

    @staticmethod
    def __calc_cv_group_delta(cv_group_size, item, g_index_to_add):
        sums = []
        for i in range(len(cv_group_size)):
            sums.append(sum(cv_group_size[i]))

        sums[g_index_to_add] += item
        return max(sums) - np.mean(sums)

    # endregion

    def _items_to_cv_pairs(self, doc_ids):
        """
        Separation with the specific separation, in terms of cv-classes size difference.
        """
        assert(isinstance(doc_ids, set))

        if not path.exists(self.__docs_stat_filepath):
            self.__docs_stat.calculate_and_write_doc_stat(filepath=self.__docs_stat_filepath,
                                                          doc_ids_iter=doc_ids)

        docs_info = self.__docs_stat.read_docs_stat(
            filepath=self.__docs_stat_filepath,
            doc_ids_set=doc_ids)

        sorted_stat = reversed(sorted(docs_info, key=lambda pair: pair[1]))
        cv_group_docs = [[] for _ in range(self.CVCount)]
        cv_group_sizes = [[] for _ in range(self.CVCount)]

        for doc_id, s_count in sorted_stat:
            g_i = self.__select_group(cv_group_size=cv_group_sizes,
                                      item=s_count)

            cv_group_docs[g_i].append(doc_id)
            cv_group_sizes[g_i].append(s_count)

        for g_index in range(len(cv_group_docs)):
            small = cv_group_docs[g_index]
            large = [doc_id for doc_id, _ in docs_info if doc_id not in small]

            yield large, small


