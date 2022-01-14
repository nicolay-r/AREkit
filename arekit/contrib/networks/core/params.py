from arekit.common.model.params import BaseModelParams


class NeuralNetworkModelParams(BaseModelParams):

    def __init__(self, epochs_count):
        super(NeuralNetworkModelParams, self).__init__()
        self.__epochs_count = epochs_count

    @property
    def EpochsCount(self):
        return self.__epochs_count
