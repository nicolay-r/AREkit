import tensorflow as tf
from base import DefaultNetworkConfig
from core.networks.context.configurations.rnn import CellTypes


class BiLSTMConfig(DefaultNetworkConfig):

    __hidden_size = 128
    __cell_type = CellTypes.BasicLSTM

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
    def DropoutRNNKeepProb(self):
        return 0.8

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    # endregion

    # region public methods

    def modify_hidden_size(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__hidden_size = value

    def modify_cell_type(self, value):
        assert(isinstance(value, unicode))
        self.__cell_type = value

    def _internal_get_parameters(self):
        parameters = super(BiLSTMConfig, self)._internal_get_parameters()

        parameters += [
            ("bi-lstm:cell_type", self.CellType),
            ("bi-lstm:hidden_size", self.HiddenSize)
        ]

        return parameters

    # endregion
