import numpy as np

from arekit.contrib.experiments.io_utils_base import BaseExperimentsIO
from arekit.contrib.experiments.cv.docs_stat import DocStatGeneratorBase


# region private methods

def __select_group(cv_group_size, item):
    deltas = []
    for i in range(len(cv_group_size)):
        delta = __calc_cv_group_delta(cv_group_size=cv_group_size,
                                      item=item,
                                      g_index_to_add=i)
        deltas.append(delta)

    return int(np.argmin(deltas))


def __calc_cv_group_delta(cv_group_size, item, g_index_to_add):
    sums = []
    for i in range(len(cv_group_size)):
        sums.append(sum(cv_group_size[i]))

    sums[g_index_to_add] += item
    return max(sums) - np.mean(sums)

# endregion


def iter_by_same_size_parts_cv(cv_count, docs_stat, experiments_io):
    """
    Separation with the specific separation, in terms of cv-classes size difference.
    """
    assert(isinstance(docs_stat, DocStatGeneratorBase))
    assert(isinstance(experiments_io, BaseExperimentsIO))

    docs_stat = docs_stat.read_docs_stat(filepath=experiments_io.get_doc_stat_filepath())
    sorted_stat = reversed(sorted(docs_stat, key=lambda pair: pair[1]))
    cv_group_docs = [[] for _ in range(cv_count)]
    cv_group_sizes = [[] for _ in range(cv_count)]

    for doc_id, s_count in sorted_stat:
        g_i = __select_group(cv_group_size=cv_group_sizes,
                             item=s_count)

        cv_group_docs[g_i].append(doc_id)
        cv_group_sizes[g_i].append(s_count)

    for g_index in range(len(cv_group_docs)):
        test = cv_group_docs[g_index]
        train = [doc_id for doc_id, _ in docs_stat
                 if doc_id not in test]

        yield train, test


def get_cv_pair_by_index(cv_count, cv_index, experiments_io, docs_stat):
    assert(isinstance(cv_count, int))
    assert(isinstance(cv_count, int) and cv_index < cv_count)
    assert(isinstance(docs_stat, DocStatGeneratorBase))
    assert(isinstance(experiments_io, BaseExperimentsIO))

    it = iter_by_same_size_parts_cv(cv_count=cv_count,
                                    experiments_io=experiments_io,
                                    docs_stat=docs_stat)

    for index, pair in enumerate(it):
        train, test = pair
        if index == cv_index:
            return train, test


