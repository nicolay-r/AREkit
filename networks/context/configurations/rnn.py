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
    __dropout_rnn_keep_prob = 1.0

    # region properties

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def HiddenSize(self):
        return self.__hidden_size

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

    def modify_cell_type(self, cell_type):
        assert(isinstance(cell_type, unicode))
        self.__cell_type = cell_type

    def modify_hidden_size(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__hidden_size = value

    def modify_dropout_rnn_keep_prob(self, value):
        assert(isinstance(value, float))
        self.__dropout_rnn_keep_prob = value

    def _internal_get_parameters(self):
        parameters = super(RNNConfig, self)._internal_get_parameters()

        parameters += [
            ("rnn:dropout_rnn_keep_prob", self.DropoutRNNKeepProb),
            ("rnn:hidden_size", self.HiddenSize),
            ("rnn:cell_type", self.CellType),
        ]

        return parameters

    # endregion
