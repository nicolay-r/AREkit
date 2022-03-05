from os import path

from enum import Enum

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class RuAttitudesVersions(Enum):
    Debug = "dbg"
    V10 = "v1_0"
    V11 = "v1_1"
    V20Base = 'v2_0_base'
    V20Large = 'v2_0_large'
    V20BaseNeut = 'v2_0_base_neut'
    V20LargeNeut = 'v2_0_large_neut'


class RuAttitudesVersionsService:

    @staticmethod
    def __iter_type_and_names():
        for version_type in RuAttitudesVersions:
            yield version_type, version_type.value

    @staticmethod
    def find_by_name(name):
        for version_type, related_name in RuAttitudesVersionsService.__iter_type_and_names():
            if name == related_name:
                return version_type
        raise Exception("Version `{}` does not supported".format(name))

    @staticmethod
    def iter_supported_names():
        for _, name in RuAttitudesVersionsService.__iter_type_and_names():
            yield name


class RuAttitudesIOUtils(ZipArchiveUtils):

    # region internal methods

    @staticmethod
    def get_archive_filepath(version):
        assert(isinstance(version, str))
        return path.join(RuAttitudesIOUtils.get_data_root(),
                         "ruattitudes-{version}.zip".format(version=version))

    @staticmethod
    def get_collection_filepath():
        return "collection.txt"

    @classmethod
    def get_synonyms_innerpath(cls):
        return "synonyms.txt"

    # endregion
