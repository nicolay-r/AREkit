import collections
import gzip

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.base import Label


class NetworkOutputEncoder(object):

    @staticmethod
    def to_tsv(filepath, sample_id_with_labels_iter, labels_scaler):
        assert(isinstance(sample_id_with_labels_iter, collections.Iterable))
        assert(isinstance(labels_scaler, BaseLabelScaler))

        col_separator = u'\t'

        with gzip.open(filepath, 'wb') as f:
            for sample_id, label in sample_id_with_labels_iter:
                assert(isinstance(label, Label))

                labels = [0] * labels_scaler.classes_count()
                labels[labels_scaler.label_to_uint(label)] = 1

                f.write(u'{s_id}{sep}{labels}\n'.format(s_id=sample_id,
                                                        sep=col_separator,
                                                        labels=col_separator.join(labels)))

