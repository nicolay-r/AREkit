import tensorflow as tf
from core.networks.context.configurations.rnn import RNNConfig
from core.networks.tf_helpers.sequence import CellTypes


class StatesAggregationModes:

    AVERAGE = u'avg'
    LAST_IN_SEQUENCE = u'last'


class IANFramesConfig(RNNConfig):

    __states_aggregation_mode = None

    def __init__(self):
        super(IANFramesConfig, self).__init__()

        # Reinitialize default parameters.
        super(IANFramesConfig, self).modify_bias_initializer(tf.zeros_initializer())
        super(IANFramesConfig, self).modify_weight_initializer(tf.random_uniform_initializer(-0.1, 0.1))
        super(IANFramesConfig, self).modify_regularizer(tf.contrib.layers.l2_regularizer(self.L2Reg))
        super(IANFramesConfig, self).modify_l2_reg(0.001)
        super(IANFramesConfig, self).modify_cell_type(CellTypes.LSTM)
        super(IANFramesConfig, self).modify_hidden_size(128)
        super(IANFramesConfig, self).modify_embedding_dropout_keep_prob(0.8)

        self.__states_aggregation_mode = StatesAggregationModes.AVERAGE

    # region Properties

    @property
    def MaxAspectLength(self):
        return self.FramesPerContext

    @property
    def MaxContextLength(self):
        return self.TermsPerContext

    @property
    def StatesAggregationMode(self):
        return self.__states_aggregation_mode

    # endregion

    # region public methods

    def modify_states_aggregation_mode(self, value):
        assert(isinstance(value, unicode))
        self.__states_aggregation_mode = value

    def _internal_get_parameters(self):
        parameters = super(IANFramesConfig, self)._internal_get_parameters()

        parameters += [
            ("ian:max_aspect_len", self.MaxAspectLength),
            ("ian:max_context_len", self.MaxContextLength),
            ("ian:aggregation_mode", self.StatesAggregationMode)
        ]

        return parameters

    # endregion
