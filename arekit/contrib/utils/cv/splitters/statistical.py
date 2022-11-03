import numpy as np
from arekit.contrib.utils.cv.doc_stat.base import BaseDocumentStatGenerator
from arekit.contrib.utils.cv.splitters.base import CrossValidationSplitter


class StatBasedCrossValidationSplitter(CrossValidationSplitter):
    """ Sentence-based splitter.
    """

    def __init__(self, docs_stat, doc_ids):
        assert(isinstance(docs_stat, BaseDocumentStatGenerator))
        super(StatBasedCrossValidationSplitter, self).__init__()
        self.__docs_info = docs_stat.calculate(doc_ids_iter=doc_ids)

    # region private methods

    @staticmethod
    def __select_group(cv_group_size, item):
        deltas = []
        for group_index in range(len(cv_group_size)):
            delta = StatBasedCrossValidationSplitter.__calc_cv_group_delta(
                cv_group_size=cv_group_size, item=item, g_index_to_add=group_index)
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
        """ Separation with the specific separation, in terms of cv-classes size difference.
        """
        assert(isinstance(doc_ids, set))
        assert(isinstance(cv_count, int))

        sorted_stat = reversed(sorted(self.__docs_info, key=lambda pair: pair[1]))
        cv_group_docs = [[] for _ in range(cv_count)]
        cv_group_sizes = [[] for _ in range(cv_count)]

        for doc_id, s_count in sorted_stat:
            group_index = self.__select_group(cv_group_size=cv_group_sizes, item=s_count)
            cv_group_docs[group_index].append(doc_id)
            cv_group_sizes[group_index].append(s_count)

        for g_index in range(len(cv_group_docs)):
            small = cv_group_docs[g_index]
            large = [doc_id for doc_id, _ in self.__docs_info if doc_id not in small]

            yield large, small
