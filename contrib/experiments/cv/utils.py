from os import path

import numpy as np

from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.doc_stat.base import DocStatGeneratorBase


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


def iter_by_same_size_parts_cv(cv_count, docs_stat, data_io):
    """
    Separation with the specific separation, in terms of cv-classes size difference.
    """
    assert(isinstance(docs_stat, DocStatGeneratorBase))
    assert(isinstance(data_io, DataIO))

    docs_stat_filepath = data_io.get_docs_stat_filepath()

    if not path.exists(docs_stat_filepath):
        docs_stat.calculate_and_write_doc_stat(docs_stat_filepath)

    docs_info = docs_stat.read_docs_stat(filepath=docs_stat_filepath)

    sorted_stat = reversed(sorted(docs_info, key=lambda pair: pair[1]))
    cv_group_docs = [[] for _ in range(cv_count)]
    cv_group_sizes = [[] for _ in range(cv_count)]

    for doc_id, s_count in sorted_stat:
        g_i = __select_group(cv_group_size=cv_group_sizes,
                             item=s_count)

        cv_group_docs[g_i].append(doc_id)
        cv_group_sizes[g_i].append(s_count)

    for g_index in range(len(cv_group_docs)):
        test = cv_group_docs[g_index]
        train = [doc_id for doc_id, _ in docs_info if doc_id not in test]

        yield train, test


def get_cv_pair_by_index(cv_count, cv_index, data_io, docs_stat):
    assert(isinstance(cv_count, int))
    assert(isinstance(cv_count, int) and cv_index < cv_count)
    assert(isinstance(docs_stat, DocStatGeneratorBase))
    assert(isinstance(data_io, DataIO))

    it = iter_by_same_size_parts_cv(cv_count=cv_count,
                                    data_io=data_io,
                                    docs_stat=docs_stat)

    for index, pair in enumerate(it):
        train, test = pair
        if index == cv_index:
            return train, test


