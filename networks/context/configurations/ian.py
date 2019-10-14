import tensorflow as tf
from base import DefaultNetworkConfig
from core.networks.context.configurations.rnn import CellTypes


class StatesAggregationModes:
    AVERAGE = u'avg'
    LAST_IN_SEQUENCE = u'last'


class IANConfig(DefaultNetworkConfig):

    __hidden_size = 128
    __aspect_len = None
    __cell_type = CellTypes.LSTM
    __dropout_rnn_keep_prob = 1.0
    __states_aggregation_mode = None

    def __init__(self):
        super(IANConfig, self).__init__()

        # Reinitialize default parameters.
        super(IANConfig, self).modify_bias_initializer(tf.zeros_initializer())
        super(IANConfig, self).modify_weight_initializer(tf.random_uniform_initializer(-0.1, 0.1))
        super(IANConfig, self).modify_optimizer(tf.train.AdamOptimizer(learning_rate=self.LearningRate))
        super(IANConfig, self).modify_regularizer(tf.contrib.layers.l2_regularizer(self.L2Reg))
        super(IANConfig, self).modify_l2_reg(0.001)

        self.__aspect_len = self.FramesPerContext
        self.__states_aggregation_mode = StatesAggregationModes.AVERAGE

    # region Properties

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    def MaxAspectLength(self):
        return self.__aspect_len

    @property
    def MaxContextLength(self):
        return self.TermsPerContext

    @property
    def DropoutRNNKeepProb(self):
        return self.__dropout_rnn_keep_prob

    @property
    def StatesAggregationMode(self):
        return self.__states_aggregation_mode

    # endregion

    # region public methods

    def modify_hidden_size(self, hidden_size):
        assert(isinstance(hidden_size, int))
        self.__hidden_size = hidden_size

    def modify_cell_type(self, cell_type):
        self.__cell_type = cell_type

    def modify_dropout_rnn_keep_prob(self, value):
        assert(isinstance(value, float))
        self.__dropout_rnn_keep_prob = value

    def modify_states_aggregation_mode(self, value):
        assert(isinstance(value, unicode))
        self.__states_aggregation_mode = value

    def _internal_get_parameters(self):
        parameters = super(IANConfig, self)._internal_get_parameters()

        parameters += [
            ("ian:cell_type", self.CellType),
            ("ian:hidden_size", self.HiddenSize),
            ("ian:max_aspect_len", self.MaxAspectLength),
            ("ian:max_context_len", self.MaxContextLength),
            ("ian:dropout_keep_prob", self.DropoutRNNKeepProb),
            ("ian:aggregation_mode", self.StatesAggregationMode)
        ]

        return parameters

    # endregion
