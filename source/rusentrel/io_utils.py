from os import path
from core.io_utils import get_data_root


class RuSentRelIOUtils(object):

    @staticmethod
    def get_filepath():
        return path.join(get_data_root(), u"rusentrel-v1_1.zip")

    @staticmethod
    def get_collection_root():
        # TODO. train, test.
        return path.join(get_data_root(), u"Collection/")

    @staticmethod
    def get_sentiment_opin_filepath(index, root, is_etalon, prefix=u'art'):
        # TODO. define root from index
        assert(isinstance(is_etalon, bool))
        return path.join(root, u"{}{}.opin{}.txt".format(prefix, index, '' if is_etalon else u'.result'))


    @staticmethod
    def get_entity_filepath(index, root):
        # TODO. define root from index
        assert(isinstance(index, int))
        assert(isinstance(root, unicode))
        return path.join(root, u"art{}.ann".format(index))


    @staticmethod
    def get_news_filepath(index, root):
        # TODO. define root from index
        assert(isinstance(index, int))
        assert(isinstance(root, unicode))
        return path.join(root, u"art{}.txt".format(index))

    @staticmethod
    def iter_test_indices():
        missed = [70]
        for i in range(46, 76):
            if i in missed:
                continue
            yield i

    @staticmethod
    def iter_train_indices():
        missed = [9, 22, 26]
        for i in range(1, 46):
            if i in missed:
                continue
            yield i

    @staticmethod
    def iter_collection_indices():
        for index in RuSentRelIOUtils.iter_train_indices():
            yield index
        for index in RuSentRelIOUtils.iter_test_indices():
            yield index

    @staticmethod
    def get_synonyms_filepath():
        return path.join(get_data_root(), u"synonyms.txt")
