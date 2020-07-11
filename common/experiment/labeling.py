import collections
import logging

logger = logging.getLogger(__name__)


class LabeledCollection:
    """
    Collection provides labeling for TextOpinionCollection
    """

    def __init__(self, labeled_sample_row_ids):
        """
        Args:
            labeled_sample_row_ids: collections.Iterable
                sequence of pairs (sample_row_id, label).
        """
        assert(isinstance(labeled_sample_row_ids, collections.Iterable))

        self.__original_labels = collections.OrderedDict(labeled_sample_row_ids)
        self.__labels_defined = {}

    def apply_label(self, label, sample_row_id):
        assert(isinstance(sample_row_id, unicode))
        assert(sample_row_id not in self.__labels_defined)
        self.__labels_defined[sample_row_id] = label

    def check_all_samples_has_labels(self):
        pass

    def check_all_samples_without_labels(self):
        return len(self.__labels_defined) == 0

    def reset_labels(self):
        self.__labels_defined.clear()
