from enum import Enum
from os.path import join, exists

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.source.rusentrel.opinions.provider import RuSentRelOpinionCollectionProvider
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding
from arekit.contrib.utils.io_utils.utils import join_dir_with_subfolder_name


def experiment_iter_index(folding):
    assert(isinstance(folding, BaseDataFolding))
    return folding.StateIndex if isinstance(folding, TwoClassCVFolding) else 0


def create_opinion_collection_target(doc_id, data_type, target_dir, labels_count, check_existance=False):
    return create_annotated_collection_target(doc_id=doc_id,
                                              data_type=data_type,
                                              check_existance=check_existance,
                                              target_dir=target_dir,
                                              labels_count=labels_count)


def create_annotated_collection_target(doc_id, data_type, target_dir, labels_count, check_existance):
    assert(isinstance(doc_id, int))
    assert(isinstance(data_type, DataType))
    assert(isinstance(check_existance, bool))
    annot_dir = __get_annotator_dir(target_dir=target_dir, labels_count=labels_count)

    if annot_dir is None:
        raise NotImplementedError("Neutral root was not provided!")

    filename = "art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id, d_type=data_type.name)

    target = join(annot_dir, filename)
    if check_existance and not exists(target):
        return None

    return target


def __get_annotator_dir(target_dir, labels_count):
    return join_dir_with_subfolder_name(dir=target_dir, subfolder_name=__get_annotator_name(labels_count))


def __get_annotator_name(labels_count):
    return "annot_{labels_count}l".format(labels_count=labels_count)


def create_result_opinion_collection_target(target_dir, doc_id, data_type, epoch_index, iteration_index):
    model_eval_root = __get_eval_root_filepath(
        target_dir=target_dir, data_type=data_type, epoch_index=epoch_index, iteration_index=iteration_index)
    return join(model_eval_root, "{}.opin.txt".format(doc_id))


def __get_eval_root_filepath(target_dir, data_type, epoch_index, iteration_index):
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    return join(target_dir, join("eval/{data_type}/{iter_index}/{epoch_index}".format(
        data_type=data_type.name, iter_index=iteration_index, epoch_index=str(epoch_index))))


def read_opinion_collection(target, labels_formatter, create_collection_func, error_on_non_supported=False):

    # Check existence of the target.
    if target is None:
        return None

    provider = RuSentRelOpinionCollectionProvider()
    opinions = provider.iter_opinions(source=target,
                                      encoding='utf-8',
                                      labels_formatter=labels_formatter,
                                      error_on_non_supported=error_on_non_supported)

    return create_collection_func(opinions)


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
