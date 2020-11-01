from os import path

from enum import Enum

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class RuAttitudesVersions(Enum):
    V10 = u"v1_0"
    V11 = u"v1_1"
    V12 = u"v1_2"
    V20Base = u'v2_0_base'
    V20Large = u'v2_0_large'


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
