import collections

from arekit.common.labels.base import Label
from arekit.common.model.labeling.base import LabelsHelper


def calculate_labels_distribution_stat(labeled_sample_row_ids, labels_helper):
    """ Provides avg. label probability stat.
    """
    assert (isinstance(labels_helper, LabelsHelper))
    assert (isinstance(labeled_sample_row_ids, collections.Iterable))

    rows_count_stat = [0] * labels_helper.get_classes_count()

    for _, label in labeled_sample_row_ids:
        assert(isinstance(label, Label))
        rows_count_stat[labels_helper.label_to_uint(label)] += 1

    total = sum(rows_count_stat)
    normalized_stat = [100.0 * value / total if total > 0 else 0 for value in rows_count_stat]
    return normalized_stat, rows_count_stat
