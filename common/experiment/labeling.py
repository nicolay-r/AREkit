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
        self._labels_defined = {}

    def apply_label(self, label, sample_row_id):
        """
        Optionally applies the label.
        Applies label if the latter has not been provided before for sample_row_id
        """
        assert(isinstance(sample_row_id, unicode))
        if sample_row_id not in self._labels_defined:
            self._labels_defined[sample_row_id] = label

    def reset_labels(self):
        self._labels_defined.clear()

    def iter_labeled_sample_row_ids(self):
        for sample_id, _ in self.__original_labels.iteritems():
            yield sample_id, self._labels_defined[sample_id]
