import tensorflow as tf
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.networks.cell_types import CellTypes


class AttentionSelfPZhouBiLSTMConfig(BiLSTMConfig):
    """
    Authors: SeoSangwoo
    Paper: https://www.aclweb.org/anthology/P16-2034
    Repository: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

    Note:
        Authors used LSTM cell by default.
    """
    __cell_type = CellTypes.LSTM

    def __init__(self):
        super(AttentionSelfPZhouBiLSTMConfig, self).__init__()

    # region properties

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    # endregion
