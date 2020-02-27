import os
import numpy as np
from os.path import join
from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIO


# region private methods

# TODO. Move into separated file
def __read_docs_stat(stat_filepath):
    """
    return:
        list of the following pairs: (doc_id, sentences_count)
    """
    docs_info = []
    with open(stat_filepath, 'r') as f:
        for line in f.readlines():
            args = [int(i) for i in line.split(':')]
            doc_id, s_count = args
            docs_info.append((doc_id, s_count))

    return docs_info


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


def get_cv_pair_by_index(cv_count, cv_index, experiments_io):
    assert(isinstance(cv_count, int))
    assert(isinstance(cv_count, int) and cv_index < cv_count)
    assert(isinstance(experiments_io, BaseExperimentsIO))

    it = iter_by_same_size_parts_cv(cv_count=cv_count,
                                    experiments_io=experiments_io)

    for index, pair in enumerate(it):
        train, test = pair
        if index == cv_index:
            return train, test


def get_path_of_subfolder_in_experiments_dir(subfolder_name, experiments_io):
    """
    Returns subfolder in experiments directory
    """
    assert(isinstance(subfolder_name, unicode))
    assert(isinstance(experiments_io, BaseExperimentsIO))

    target_dir = join(experiments_io.get_experiments_dir(), u"{}/".format(subfolder_name))
    create_dir_if_not_exists(target_dir)
    return target_dir


# TODO. To base io. Refactoring
def get_rusentrel_stats_filepath(experiments_io):
    assert(isinstance(experiments_io, BaseExperimentsIO))
    return os.path.join(experiments_io.get_data_root(), u"rusentrel_docs_stat.txt")


def iter_by_same_size_parts_cv(cv_count, experiments_io):
    """
    Separation with the specific separation, in terms of cv-classes size difference.
    """
    assert(isinstance(experiments_io, BaseExperimentsIO))

    stat = __read_docs_stat(stat_filepath=experiments_io.get_rusentrel_stats_filepath())
    sorted_stat = reversed(sorted(stat, key=lambda pair: pair[1]))
    cv_group_docs = [[] for _ in range(cv_count)]
    cv_group_sizes = [[] for _ in range(cv_count)]

    for doc_id, s_count in sorted_stat:
        g_i = __select_group(cv_group_size=cv_group_sizes,
                             item=s_count)

        cv_group_docs[g_i].append(doc_id)
        cv_group_sizes[g_i].append(s_count)

    for g_index in range(len(cv_group_docs)):
        test = cv_group_docs[g_index]
        train = [doc_id for doc_id, _ in stat
                 if doc_id not in test]

        yield train, test


