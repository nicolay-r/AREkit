import collections
import gzip

from arekit.common.experiment import const
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.base import Label
from arekit.common.utils import create_dir_if_not_exists


class NetworkOutputEncoder(object):

    @staticmethod
    def to_tsv(filepath, sample_id_with_labels_iter, labels_scaler, col_separator=u'\t'):
        assert(isinstance(sample_id_with_labels_iter, collections.Iterable))
        assert(isinstance(labels_scaler, BaseLabelScaler))

        create_dir_if_not_exists(filepath)

        with gzip.open(filepath, 'wb') as f:

            # Writing title.
            title = [const.ID]
            title.extend([unicode(labels_scaler.label_to_uint(label))
                          for label in labels_scaler.ordered_suppoted_labels()])
            f.write(u"{}\n".format(col_separator.join(title)))

            # Writing contents.
            for sample_id, label in sample_id_with_labels_iter:
                assert(isinstance(label, Label))

                labels = [u'0'] * labels_scaler.classes_count()
                labels[labels_scaler.label_to_uint(label)] = u'1'

                f.write(u'{s_id}{sep}{labels}\n'.format(
                    s_id=sample_id,
                    sep=col_separator,
                    labels=col_separator.join(labels)))
