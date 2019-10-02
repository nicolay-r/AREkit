import tensorflow as tf
from base import DefaultNetworkConfig


class CellTypes:
    RNN = u'vanilla'
    GRU = u'gru'
    LSTM = u'lstm'
    BasicLSTM = u'basic-lstm'


class RNNConfig(DefaultNetworkConfig):

    __hidden_size = 300
    __cell_type = CellTypes.BasicLSTM

    # region properties

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    def LearningRate(self):
        return 0.01

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    # endregion

    # region public methods

    def set_cell_type(self, cell_type):
        assert(isinstance(cell_type, unicode))
        self.__cell_type = cell_type

    def modify_hidden_size(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__hidden_size = value

    def _internal_get_parameters(self):
        parameters = super(RNNConfig, self)._internal_get_parameters()

        parameters += [
            ("rnn:hidden_size", self.HiddenSize),
            ("rnn:cell_type", self.CellType),
        ]

        return parameters

    # endregion
