from os import path
from core.io_utils import get_data_root


# TODO. Use this io in reader, i.e. inside core (not outside)!
class RuSentiFramesIO(object):

    @staticmethod
    def get_filepath():
        return path.join(get_data_root(), u"rusentiframes-v1_0.zip")
