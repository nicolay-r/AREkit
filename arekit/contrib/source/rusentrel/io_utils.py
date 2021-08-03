from os import path

from enum import Enum

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class RuSentRelVersions(Enum):
    V11 = u"v1_1"


class RuSentRelIOUtils(ZipArchiveUtils):

    __sep_doc_id = 46

    @staticmethod
    def get_archive_filepath(version):
        assert(version, unicode)
        return path.join(RuSentRelIOUtils.get_data_root(), u"rusentrel-{}.zip".format(version))

    # region internal methods

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

    @staticmethod
    def __is_supported(version):
        assert(isinstance(version, RuSentRelVersions))
        if version != RuSentRelVersions.V11:
            raise NotImplementedError("Collection does not supported")
        return True

    @staticmethod
    def __number_from_string(s):
        digit_chars = [chr for chr in s if chr.isdigit()]

        if len(digit_chars) == 0:
            return None

        return int(u"".join(digit_chars))

    @staticmethod
    def __iter_indicies_from_dataset(version, folder_name):
        assert(isinstance(folder_name, unicode))
        assert(RuSentRelIOUtils.__is_supported(version))

        used = set()

        for filename in RuSentRelIOUtils.iter_filenames_from_zip(version):
            if not folder_name in filename:
                continue

            index = RuSentRelIOUtils.__number_from_string(filename)

            if index is None:
                continue

            if index in used:
                continue

            used.add(index)

            yield index

    # region public methods

    @staticmethod
    def iter_test_indices(version):
        assert(RuSentRelIOUtils.__is_supported(version))
        for index in RuSentRelIOUtils.__iter_indicies_from_dataset(version=version, folder_name=u"test/"):
            yield index

    @staticmethod
    def iter_train_indices(version):
        assert(RuSentRelIOUtils.__is_supported(version))
        for index in RuSentRelIOUtils.__iter_indicies_from_dataset(version=version, folder_name=u"train/"):
            yield index

    @staticmethod
    def iter_collection_indices(version):
        assert(RuSentRelIOUtils.__is_supported(version))
        for index in RuSentRelIOUtils.iter_train_indices(version):
            yield index
        for index in RuSentRelIOUtils.iter_test_indices(version):
            yield index

    # endregion
