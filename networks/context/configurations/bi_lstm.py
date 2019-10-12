import tensorflow as tf
from base import DefaultNetworkConfig
from core.networks.context.configurations.rnn import CellTypes


class BiLSTMConfig(DefaultNetworkConfig):

    __hidden_size = 128
    __cell_type = CellTypes.BasicLSTM
    __dropout_rnn_keep_prob = 1.0

    # region properties

    @property
    def L2Reg(self):
        return 0.001

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def LearningRate(self):
        return 0.1

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    @property
    def DropoutRNNKeepProb(self):
        return self.__dropout_rnn_keep_prob

    # endregion

    # region public methods

    def modify_hidden_size(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__hidden_size = value

    def modify_cell_type(self, value):
        assert(isinstance(value, unicode))
        self.__cell_type = value

    def modify_dropout_rnn_keep_prob(self, value):
        assert(isinstance(value, float))
        self.__dropout_rnn_keep_prob = value

    def _internal_get_parameters(self):
        parameters = super(BiLSTMConfig, self)._internal_get_parameters()

        parameters += [
            ("bi-lstm:dropout_rnn_keep_prob", self.DropoutRNNKeepProb),
            ("bi-lstm:cell_type", self.CellType),
            ("bi-lstm:hidden_size", self.HiddenSize)
        ]

        return parameters

    # endregion
