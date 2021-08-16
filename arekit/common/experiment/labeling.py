import collections
import logging

logger = logging.getLogger(__name__)


class LabeledCollection:
    """
    Collection provides labeling for TextOpinionCollection
    """

    def __init__(self, uint_labeled_sample_row_ids):
        """
        Args:
            uint_labeled_sample_row_ids: collections.Iterable
                sequence of pairs (sample_row_id, uint_label).
        """
        assert(isinstance(uint_labeled_sample_row_ids, collections.Iterable))

        self.__original_labels_by_row_id_dict = collections.OrderedDict(uint_labeled_sample_row_ids)
        self._uint_labels_defined = {}

    def is_empty(self):
        return len(self._uint_labels_defined) == 0

    def apply_uint_label(self, uint_label, sample_row_id):
        """
        Optionally applies the label.
        Applies label if the latter has not been provided before for sample_row_id
        """
        assert(isinstance(uint_label, int))
        assert(isinstance(sample_row_id, str))

        if sample_row_id not in self._uint_labels_defined:
            self._uint_labels_defined[sample_row_id] = uint_label

    def reset_labels(self):
        self._uint_labels_defined.clear()

    def iter_non_duplicated_labeled_sample_row_ids(self):
        for sample_id, _ in self.__original_labels_by_row_id_dict.items():
            yield sample_id, self._uint_labels_defined[sample_id]
