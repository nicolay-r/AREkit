from os import path

from enum import Enum

from arekit.source.zip_utils import ZipArchiveUtils


class RuSentiFramesVersions(Enum):

    # Papers for description:
    # Distant Supervision for Sentiment Attitude Extraction (RANLP-2019)
    # Nicolay Rusnachenko, Natalia Loukachevitch, Elena Tutubalina
    # https://www.aclweb.org/anthology/R19-1118/
    # https://github.com/nicolay-r/RuSentiFrames/tree/v1.0
    V10 = u"v1_0"

    # Papers for description:
    # Sentiment Frames for Attitude Extraction in Russian (DIALOG-2020)
    # Natalia Loukachevitch, Nicolay Rusnachenko
    # https://github.com/nicolay-r/RuSentiFrames/tree/v2.0
    V20 = u"v2_0"


class RuSentiFramesIOUtils(ZipArchiveUtils):

    # region internal methods

    @staticmethod
    def get_archive_filepath(version):
        assert(isinstance(version, unicode))
        return path.join(RuSentiFramesIOUtils.get_data_root(), u"rusentiframes-{version}.zip".format(version=version))

    @staticmethod
    def get_collection_filepath():
        return u"frames.json"

    # endregion
