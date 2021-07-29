import collections
import gzip

from arekit.common.experiment import const
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.utils import create_dir_if_not_exists


class NetworkOutputEncoder(object):

    @staticmethod
    def to_tsv(filepath, sample_id_with_uint_labels_iter, labels_scaler,
               column_extra_funcs=None, col_separator=u'\t'):
        assert(isinstance(sample_id_with_uint_labels_iter, collections.Iterable))
        assert(isinstance(labels_scaler, BaseLabelScaler))
        assert(isinstance(column_extra_funcs, list) or column_extra_funcs is None)

        create_dir_if_not_exists(filepath)

        with gzip.open(filepath, 'wb') as f:

            # Writing title.
            title = [const.ID]
            title.extend([column_name for column_name, _ in column_extra_funcs])
            title.extend([unicode(labels_scaler.label_to_uint(label))
                          for label in labels_scaler.ordered_suppoted_labels()])
            f.write(u"{}\n".format(col_separator.join(title)))

            # Writing contents.
            for sample_id, uint_label in sample_id_with_uint_labels_iter:
                assert(isinstance(uint_label, int))

                labels = [u'0'] * labels_scaler.classes_count()
                labels[uint_label] = u'1'

                # Composing row contents.
                contents = [sample_id]

                # Optionally provide additional values.
                if column_extra_funcs is not None:
                    for _, value_func in column_extra_funcs:
                        contents.append(str(value_func(sample_id)))

                # Providing row labels.
                contents.extend(labels)

                # Saving the related row.
                f.write(u'{}\n'.format(col_separator.join(contents)))
