import tensorflow as tf
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.networks.cell_types import CellTypes


class AttentionSelfPZhouRCNNConfig(RCNNConfig):

    __cell_type = CellTypes.LSTM

    def __init__(self):
        super(AttentionSelfPZhouRCNNConfig, self).__init__()

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