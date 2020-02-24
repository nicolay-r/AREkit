import tensorflow as tf
from arekit.networks.context.configurations.rnn import RNNConfig
from arekit.networks.tf_helpers.sequence import CellTypes


class BiLSTMConfig(RNNConfig):

    __lstm_cell_initializer = tf.keras.initializers.glorot_normal()

    def __init__(self):
        super(BiLSTMConfig, self).__init__()
        super(BiLSTMConfig, self).modify_hidden_size(128)
        super(BiLSTMConfig, self).modify_l2_reg(0.001)
        super(BiLSTMConfig, self).modify_dropout_rnn_keep_prob(0.8)
        super(BiLSTMConfig, self).modify_cell_type(CellTypes.BasicLSTM)
        super(BiLSTMConfig, self).modify_bias_initializer(tf.constant_initializer(0.1))
        super(BiLSTMConfig, self).modify_weight_initializer(tf.contrib.layers.xavier_initializer())

    # region properties

    @property
    def LSTMCellInitializer(self):
        return self.__lstm_cell_initializer

    # endregion
