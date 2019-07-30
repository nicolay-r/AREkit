from os import path
from core.io_utils import get_data_root


# TODO. Use this io in reader, i.e. inside core (not outside)!
class RuAttitudesIO(object):

    # region internal methods

    @staticmethod
    def get_filepath():
        return path.join(get_data_root(), u"ruattitudes-v1_0.zip")

    @staticmethod
    def get_collection_filepath():
        return u"collection.txt"

    # endregion
