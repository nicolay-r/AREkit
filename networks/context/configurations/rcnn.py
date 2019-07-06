from base import CommonModelSettings
from core.networks.context.configurations import CellTypes


class RCNNConfig(CommonModelSettings):

    __hidden_size = 128
    __context_embedding_size = 300
    __cell_type = CellTypes.LSTM

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def SurroundingOneSideContextEmbeddingSize(self):
        return self.__context_embedding_size

    @property
    def HiddenSize(self):
        return self.__hidden_size

    def _internal_get_parameters(self):
        parameters = super(RCNNConfig, self)._internal_get_parameters()

        parameters += [
            ("rcnn:cell_type", self.CellType),
            ("rcnn:hidden_size", self.HiddenSize),
            ("rcnn:context_size (one side)", self.SurroundingOneSideContextEmbeddingSize),
        ]

        return parameters
