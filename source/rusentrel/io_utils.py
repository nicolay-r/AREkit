import zipfile
from os import path
from core.io_utils import get_data_root
from core.source.base import BaseIOUtils


class RuSentRelIOUtils(BaseIOUtils):

    __sep_doc_id = 46

    @staticmethod
    def get_archive_filepath():
        return path.join(get_data_root(), u"rusentrel-v1_1.zip")

    # region internal methods

    @staticmethod
    def read_from_zip(inner_path, process_func):
        """
        process_func:
            func which receives a file reader
        """
        assert(isinstance(inner_path, unicode))
        assert(callable(process_func))

        with zipfile.ZipFile(RuSentRelIOUtils.get_archive_filepath(), "r") as zip_ref:
            with zip_ref.open(inner_path, mode='r') as c_file:
                return process_func(c_file)

    @staticmethod
    def get_sentiment_opin_filepath(index, prefix=u'art'):
        root = RuSentRelIOUtils.__get_root_by_index(index, is_opinion=True)
        return path.join(root, u"{}{}.opin.txt".format(prefix, index))

    @staticmethod
    def get_entity_innerpath(index):
        assert(isinstance(index, int))
        inner_root = RuSentRelIOUtils.__get_root_by_index(index)
        return path.join(inner_root, u"art{}.ann".format(index))

    @staticmethod
    def get_news_innerpath(index):
        assert(isinstance(index, int))
        inner_root = RuSentRelIOUtils.__get_root_by_index(index)
        return path.join(inner_root, u"art{}.txt".format(index))

    @staticmethod
    def get_synonyms_innerpath():
        return u"synonyms.txt"

    # endregion

    @staticmethod
    def __get_root_by_index(doc_id, is_opinion=False):
        assert(isinstance(doc_id, int))
        other_dir = u'etalon' if is_opinion else u'test'
        return other_dir if doc_id >= RuSentRelIOUtils.__sep_doc_id else u"train"

    # region public methods

    @staticmethod
    def iter_test_indices():
        missed = [70]
        for i in range(RuSentRelIOUtils.__sep_doc_id, 76):
            if i in missed:
                continue
            yield i

    @staticmethod
    def iter_train_indices():
        missed = [9, 22, 26]
        for i in range(1, RuSentRelIOUtils.__sep_doc_id):
            if i in missed:
                continue
            yield i

    @staticmethod
    def iter_collection_indices():
        for index in RuSentRelIOUtils.iter_train_indices():
            yield index
        for index in RuSentRelIOUtils.iter_test_indices():
            yield index

    # endregion
