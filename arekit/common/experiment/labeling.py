import collections
import logging

logger = logging.getLogger(__name__)


class LabeledCollection:

    def __init__(self, uint_labeled_ids):
        """
        Args:
            uint_labeled_ids: collections.Iterable
                sequence of pairs (sample_row_id, uint_label).
        """
        assert(isinstance(uint_labeled_ids, collections.Iterable))

        self.__original_uint_labels = collections.OrderedDict(uint_labeled_ids)
        self.__assigned_uint_labels = {}

    def assign_uint_label(self, uint_label, sample_row_id):
        """
        Optionally applies the label.
        Applies label if the latter has not been provided before for sample_row_id
        """
        assert(isinstance(uint_label, int))
        assert(isinstance(sample_row_id, str))

        if sample_row_id not in self.__assigned_uint_labels:
            self.__assigned_uint_labels[sample_row_id] = uint_label

    def iter_non_duplicated_labeled_sample_row_ids(self):
        for sample_id, _ in self.__original_uint_labels.items():
            yield sample_id, self.__assigned_uint_labels[sample_id]
