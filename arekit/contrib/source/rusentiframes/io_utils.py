from os import path

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class RuSentiFramesIOUtils(ZipArchiveUtils):

    # region internal methods

    @staticmethod
    def get_archive_filepath(version):
        assert(isinstance(version, str))
        return path.join(RuSentiFramesIOUtils.get_data_root(), "rusentiframes-{version}.zip".format(version=version))

    @staticmethod
    def get_collection_filepath():
        return "frames.json"

    # endregion
