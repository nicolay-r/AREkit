from os import path

import numpy as np

from arekit.contrib.experiments.cv.base import BaseCVFolding
from arekit.contrib.experiments.doc_stat.base import DocStatGeneratorBase


class SentenceBasedCVFolding(BaseCVFolding):
    """
    Sentence-based separation.
    Considering a separation in foldings, equal each other in terms of sentence count.
    """

    def __init__(self, docs_stat, docs_stat_filepath):
        assert(isinstance(docs_stat, DocStatGeneratorBase))
        assert(isinstance(docs_stat_filepath, unicode))
        super(SentenceBasedCVFolding, self).__init__()

        self.__docs_stat = docs_stat
        self.__docs_stat_filepath = docs_stat_filepath

    # region private methods

    @staticmethod
    def __select_group(cv_group_size, item):
        deltas = []
        for i in range(len(cv_group_size)):
            delta = SentenceBasedCVFolding.__calc_cv_group_delta(
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

    def __iter_by_same_size_parts_cv(self):
        """
        Separation with the specific separation, in terms of cv-classes size difference.
        """

        if not path.exists(self.__docs_stat_filepath):
            self.__docs_stat.calculate_and_write_doc_stat(self.__docs_stat_filepath)

        docs_info = self.__docs_stat.read_docs_stat(filepath=self.__docs_stat_filepath)

        sorted_stat = reversed(sorted(docs_info, key=lambda pair: pair[1]))
        cv_group_docs = [[] for _ in range(self.CVCount)]
        cv_group_sizes = [[] for _ in range(self.CVCount)]

        for doc_id, s_count in sorted_stat:
            g_i = self.__select_group(cv_group_size=cv_group_sizes,
                                      item=s_count)

            cv_group_docs[g_i].append(doc_id)
            cv_group_sizes[g_i].append(s_count)

        for g_index in range(len(cv_group_docs)):
            test = cv_group_docs[g_index]
            train = [doc_id for doc_id, _ in docs_info if doc_id not in test]

            yield train, test

    # endregion

    def get_cv_pair_by_index(self):

        it = self.__iter_by_same_size_parts_cv()

        for index, pair in enumerate(it):
            train, test = pair
            if index == self.IterationIndex:
                return train, test


