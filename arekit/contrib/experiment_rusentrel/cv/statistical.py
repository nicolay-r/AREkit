from os import path
import numpy as np
from arekit.common.experiment.cv.splitters.base import CrossValidationSplitter
from arekit.contrib.experiment_rusentrel.cv.doc_stat.base import BaseDocumentStatGenerator


class StatBasedCrossValidataionSplitter(CrossValidationSplitter):
    """ Sentence-based splitter.
    """

    def __init__(self, docs_stat, docs_stat_filepath_func):
        assert(isinstance(docs_stat, BaseDocumentStatGenerator))
        assert(callable(docs_stat_filepath_func))
        super(StatBasedCrossValidataionSplitter, self).__init__()

        self.__docs_stat = docs_stat
        self.__docs_stat_filepath_func = docs_stat_filepath_func

    # region private methods

    @staticmethod
    def __select_group(cv_group_size, item):
        deltas = []
        for i in range(len(cv_group_size)):
            delta = StatBasedCrossValidataionSplitter.__calc_cv_group_delta(
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

    def items_to_cv_pairs(self, doc_ids, cv_count):
        """
        Separation with the specific separation, in terms of cv-classes size difference.
        """
        assert(isinstance(doc_ids, set))
        assert(isinstance(cv_count, int))

        filepath = self.__docs_stat_filepath_func()

        if not path.exists(filepath):
            self.__docs_stat.calculate_and_write_doc_stat(filepath=filepath,
                                                          doc_ids_iter=doc_ids)

        docs_info = self.__docs_stat.read_docs_stat(filepath=filepath,
                                                    doc_ids_set=doc_ids)

        sorted_stat = reversed(sorted(docs_info, key=lambda pair: pair[1]))
        cv_group_docs = [[] for _ in range(cv_count)]
        cv_group_sizes = [[] for _ in range(cv_count)]

        for doc_id, s_count in sorted_stat:
            g_i = self.__select_group(cv_group_size=cv_group_sizes,
                                      item=s_count)

            cv_group_docs[g_i].append(doc_id)
            cv_group_sizes[g_i].append(s_count)

        for g_index in range(len(cv_group_docs)):
            small = cv_group_docs[g_index]
            large = [doc_id for doc_id, _ in docs_info if doc_id not in small]

            yield large, small
