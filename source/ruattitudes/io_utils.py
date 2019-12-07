from os import path
from arekit.source.base import BaseIOUtils


class RuAttitudesIOUtils(BaseIOUtils):

    # region internal methods

    @staticmethod
    def get_archive_filepath():
        return path.join(RuAttitudesIOUtils.get_data_root(), u"ruattitudes-v1_1.zip")

    @staticmethod
    def get_collection_filepath():
        return u"collection.txt"

    # endregion
