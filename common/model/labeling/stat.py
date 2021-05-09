import collections


def calculate_labels_distribution_stat(uint_labeled_sample_row_ids, classes_count):
    """ Provides avg. label probability stat.
    """
    assert(isinstance(classes_count, int))
    assert(isinstance(uint_labeled_sample_row_ids, collections.Iterable))

    rows_count_stat = [0] * classes_count

    for _, uint_label in uint_labeled_sample_row_ids:
        rows_count_stat[uint_label] += 1

    total = sum(rows_count_stat)
    normalized_stat = [100.0 * value / total if total > 0 else 0 for value in rows_count_stat]
    return normalized_stat, rows_count_stat
