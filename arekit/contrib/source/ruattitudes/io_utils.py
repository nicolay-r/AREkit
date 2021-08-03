from os import path

from enum import Enum

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class RuAttitudesVersions(Enum):
    Debug = u"dbg"
    V10 = u"v1_0"
    V11 = u"v1_1"
    V12 = u"v1_2"
    V20Base = u'v2_0_base'
    V20Large = u'v2_0_large'
    V20BaseNeut = u'v2_0_base_neut'
    V20LargeNeut = u'v2_0_large_neut'


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
        assert(isinstance(version, unicode))
        return path.join(RuAttitudesIOUtils.get_data_root(),
                         u"ruattitudes-{version}.zip".format(version=version))

    @staticmethod
    def get_collection_filepath():
        return u"collection.txt"

    @classmethod
    def get_synonyms_innerpath(cls):
        return u"synonyms.txt"

    # endregion
