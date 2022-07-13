from enum import Enum
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding


def experiment_iter_index(folding):
    assert(isinstance(folding, BaseDataFolding))
    return folding.StateIndex if isinstance(folding, TwoClassCVFolding) else 0


def create_result_opinion_collection_target(target_dir, doc_id, data_type, epoch_index, iteration_index):
    model_eval_root = __get_eval_root_filepath(
        target_dir=target_dir, data_type=data_type, epoch_index=epoch_index, iteration_index=iteration_index)
    return join(model_eval_root, "{}.opin.txt".format(doc_id))


def __get_eval_root_filepath(target_dir, data_type, epoch_index, iteration_index):
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    return join(target_dir, join("eval/{data_type}/{iter_index}/{epoch_index}".format(
        data_type=data_type.name, iter_index=iteration_index, epoch_index=str(epoch_index))))


class EnumConversionService(object):

    _data = None

    @classmethod
    def is_supported(cls, name):
        assert(isinstance(cls._data, dict))
        return name in cls._data

    @classmethod
    def name_to_type(cls, name):
        assert(isinstance(cls._data, dict))
        assert(isinstance(name, str))
        return cls._data[name]

    @classmethod
    def iter_names(cls):
        assert(isinstance(cls._data, dict))
        return iter(list(cls._data.keys()))

    @classmethod
    def type_to_name(cls, enum_type):
        assert(isinstance(cls._data, dict))
        assert(isinstance(enum_type, Enum))

        for item_name, item_type in cls._data.items():
            if item_type == enum_type:
                return item_name

        raise NotImplemented("Formatting type '{}' does not supported".format(enum_type))
