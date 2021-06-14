from arekit.common.model.params import BaseModelParams


class NeuralNetworkModelParams(BaseModelParams):

    def __init__(self, epochs_count):
        self.__epochs_count = epochs_count

    @property
    def EpochsCount(self):
        return self.__epochs_count
