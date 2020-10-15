from os import path
from arekit.source.zip_utils import ZipArchiveUtils


class RuAttitudesVersions:
    V10 = u"v1_0"
    V11 = u"v1_1"
    V20Base = u"v2_0_base"
    V20Large = u"v2_0_large"


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

    # endregion
