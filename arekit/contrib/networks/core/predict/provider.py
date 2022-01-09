import collections

from arekit.common.data import const
from arekit.common.labels.scaler.base import BaseLabelScaler


class BasePredictProvider(object):

    @staticmethod
    def __iter_contents(sample_id_with_uint_labels_iter, labels_scaler, column_extra_funcs):

        for sample_id, uint_label in sample_id_with_uint_labels_iter:
            assert(isinstance(uint_label, int))

            labels = ['0'] * labels_scaler.classes_count()
            labels[uint_label] = '1'

            # Composing row contents.
            contents = [sample_id]

            # Optionally provide additional values.
            if column_extra_funcs is not None:
                for _, value_func in column_extra_funcs:
                    contents.append(str(value_func(sample_id)))

            # Providing row labels.
            contents.extend(labels)
            yield contents

    def provide(self, sample_id_with_uint_labels_iter, labels_scaler, column_extra_funcs=None):
        assert(isinstance(sample_id_with_uint_labels_iter, collections.Iterable))
        assert(isinstance(labels_scaler, BaseLabelScaler))
        assert(isinstance(column_extra_funcs, list) or column_extra_funcs is None)

        # Provide contents.
        contents_it = self.__iter_contents(
            sample_id_with_uint_labels_iter=sample_id_with_uint_labels_iter,
            labels_scaler=labels_scaler,
            column_extra_funcs=column_extra_funcs)

        # Provide title.
        title = [const.ID]
        if column_extra_funcs is not None:
            title.extend([column_name for column_name, _ in column_extra_funcs])
        title.extend([str(labels_scaler.label_to_uint(label))
                      for label in labels_scaler.ordered_suppoted_labels()])

        return title, contents_it
