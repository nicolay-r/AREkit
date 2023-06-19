from os import path

from enum import Enum

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class RuSentRelVersions(Enum):
    """ Original collection repository: https://github.com/nicolay-r/RuSentRel
        Paper: https://arxiv.org/abs/1808.08932
    """
    V11 = "v1_1"


class RuSentRelIOUtils(ZipArchiveUtils):

    TEST_FOLDER = "test"
    TRAIN_FOLDER = "train"
    ETALON_FOLDER = "etalon"

    @staticmethod
    def get_archive_filepath(version):
        assert(version, str)
        return path.join(RuSentRelIOUtils.get_data_root(), "rusentrel-{}.zip".format(version))

    # region internal methods

    @staticmethod
    def get_sentiment_opin_filepath(index, version, prefix='art'):
        root = RuSentRelIOUtils.__get_root_by_index(index, version=version, keep_etalon=True)
        return path.join(root, "{prefix}{index}.opin.txt".format(prefix=prefix, index=index))

    @staticmethod
    def get_entity_innerpath(index, version):
        assert(isinstance(index, int))
        assert(isinstance(version, RuSentRelVersions))
        inner_root = RuSentRelIOUtils.__get_root_by_index(doc_id=index, version=version)
        return path.join(inner_root, "art{}.ann".format(index))

    @staticmethod
    def get_doc_innerpath(index, version):
        assert(isinstance(index, int))
        assert(isinstance(version, RuSentRelVersions))
        inner_root = RuSentRelIOUtils.__get_root_by_index(doc_id=index, version=version)
        return path.join(inner_root, "art{}.txt".format(index))

    @staticmethod
    def get_synonyms_innerpath():
        return "synonyms.txt"

    # endregion

    @staticmethod
    def __get_root_by_index(doc_id, version, keep_etalon=False):
        assert(RuSentRelIOUtils.__is_supported(version))
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(doc_id, int))
        other_dir = RuSentRelIOUtils.ETALON_FOLDER if keep_etalon else RuSentRelIOUtils.TEST_FOLDER
        test_indices = set(RuSentRelIOUtils.__iter_indicies_from_dataset(version, RuSentRelIOUtils.TEST_FOLDER))
        return other_dir if doc_id in test_indices else RuSentRelIOUtils.TRAIN_FOLDER

    @staticmethod
    def __is_supported(version):
        assert(isinstance(version, RuSentRelVersions))
        return version == RuSentRelVersions.V11

    @staticmethod
    def __number_from_string(s):
        digit_chars = [chr for chr in s if chr.isdigit()]

        if len(digit_chars) == 0:
            return None

        return int("".join(digit_chars))

    @staticmethod
    def __iter_indicies_from_dataset(version, folder_name):
        assert(isinstance(folder_name, str))
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
        indices_iter = RuSentRelIOUtils.__iter_indicies_from_dataset(
            version=version, folder_name="{}/".format(RuSentRelIOUtils.TEST_FOLDER))
        for index in indices_iter:
            yield index

    @staticmethod
    def iter_train_indices(version):
        assert(RuSentRelIOUtils.__is_supported(version))
        indices_iter = RuSentRelIOUtils.__iter_indicies_from_dataset(
            version=version, folder_name="{}/".format(RuSentRelIOUtils.TRAIN_FOLDER))
        for index in indices_iter:
            yield index

    @staticmethod
    def iter_collection_indices(version):
        assert(RuSentRelIOUtils.__is_supported(version))
        for index in RuSentRelIOUtils.iter_train_indices(version):
            yield index
        for index in RuSentRelIOUtils.iter_test_indices(version):
            yield index

    # endregion
