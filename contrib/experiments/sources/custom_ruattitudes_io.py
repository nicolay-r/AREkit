from arekit.networks.network_io import NetworkIO
from arekit.source.ruattitudes.io_utils import RuAttitudesIOUtils


class CustomRuAttitudesIOUtils(RuAttitudesIOUtils):
    pass


class CustomRuAttitudesFormatIO(NetworkIO):
    # TODO. Implement (like in rusentrel_with_ruattitudes_io)

    def __init__(self):
        self.__ru_attitudes_test = None
