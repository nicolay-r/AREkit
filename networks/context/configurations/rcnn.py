import tensorflow as tf
from base import DefaultNetworkConfig
from core.networks.context.configurations.rnn import CellTypes


class RCNNConfig(DefaultNetworkConfig):

    __hidden_size = 128
    __context_embedding_size = 300
    __cell_type = CellTypes.BasicLSTM
    __dropout_rnn_keep_prob = 1.0

    # region properties

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def SurroundingOneSideContextEmbeddingSize(self):
        return self.__context_embedding_size

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def DropoutRNNKeepProb(self):
        return self.__dropout_rnn_keep_prob

    # endregion

    def modify_cell_type(self, value):
        self.__cell_type = value

    def modify_dropout_rnn_keep_prob(self, value):
        assert(isinstance(value, float))
        self.__dropout_rnn_keep_prob = value

    def _internal_get_parameters(self):
        parameters = super(RCNNConfig, self)._internal_get_parameters()

        parameters += [
            ("rcnn:dropout_rnn_keep_prob", self.DropoutRNNKeepProb),
            ("rcnn:cell_type", self.CellType),
            ("rcnn:hidden_size", self.HiddenSize),
            ("rcnn:context_size (one side)", self.SurroundingOneSideContextEmbeddingSize),
        ]

        return parameters
