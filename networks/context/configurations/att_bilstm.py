import tensorflow as tf
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.tf_helpers.sequence import CellTypes


class AttBiLSTMConfig(BiLSTMConfig):
    """
    Authors: SeoSangwoo
    Paper: https://www.aclweb.org/anthology/P16-2034
    Repository: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

    Note:
        Authors used LSTM cell by default.
    """
    __cell_type = CellTypes.LSTM
    __lstm_cell_initializer = tf.keras.initializers.glorot_normal()

    def __init__(self):
        super(AttBiLSTMConfig, self).__init__()

    # region properties

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def LSTMCellInitializer(self):
        return self.__lstm_cell_initializer

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    # endregion
