import tensorflow as tf
from core.networks.context.configurations.rnn import RNNConfig
from core.networks.tf_helpers.sequence import CellTypes


class RCNNConfig(RNNConfig):

    __context_embedding_size = 300

    def __init__(self):
        super(RCNNConfig, self).__init__()
        super(RCNNConfig, self).modify_hidden_size(128)
        super(RCNNConfig, self).modify_cell_type(CellTypes.BasicLSTM)
        super(RCNNConfig, self).modify_dropout_rnn_keep_prob(0.8)

    # region properties

    @property
    def SurroundingOneSideContextEmbeddingSize(self):
        return self.__context_embedding_size

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    # endregion

    def _internal_get_parameters(self):
        parameters = super(RCNNConfig, self)._internal_get_parameters()

        parameters += [
            ("rcnn:context_size (one side)", self.SurroundingOneSideContextEmbeddingSize),
        ]

        return parameters
