from os import path
from arekit.source.base import BaseIOUtils


class RuSentiFramesIOUtils(BaseIOUtils):

    # region internal methods

    @staticmethod
    def get_archive_filepath():
        return path.join(RuSentiFramesIOUtils.get_data_root(), u"rusentiframes-v1_0.zip")

    @staticmethod
    def get_collection_filepath():
        return u"frames.json"

    # endregion
