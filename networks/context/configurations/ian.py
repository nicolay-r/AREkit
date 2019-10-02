import tensorflow as tf
from base import DefaultNetworkConfig


class IANConfig(DefaultNetworkConfig):

    __hidden_size = 128
    __aspect_len = None
    __aspect_embedding_matrix = None

    def __init__(self):
        super(IANConfig, self).__init__()
        self.__aspect_len = self.FramesPerContext

    # region Properties

    @property
    def L2Reg(self):
        return 0.001

    @property
    def LearningRate(self):
        return 0.01

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
    def AspectsEmbedding(self):
        return self.__aspect_embedding_matrix

    @property
    def BiasInitializer(self):
        return tf.zeros_initializer()

    @property
    def WeightInitializer(self):
        return tf.random_uniform_initializer(-0.1, 0.1)

    @property
    def AspectsEmbeddingShape(self):
        return self.__aspect_embedding_matrix.shape

    @property
    def Optimiser(self):
        return tf.train.AdamOptimizer(learning_rate=self.LearningRate)

    @property
    def LayerRegularizer(self):
        return tf.contrib.layers.l2_regularizer(self.L2Reg)

    # endregion

    # region public methods

    def modify_hidden_size(self, hidden_size):
        assert(isinstance(hidden_size, int))
        self.__hidden_size = hidden_size

    def notify_initialization_completed(self):
        self.__aspect_embedding_matrix = self.TermEmbeddingMatrix

    def _internal_get_parameters(self):
        parameters = super(IANConfig, self)._internal_get_parameters()

        parameters += [
            ("ian:hidden_size", self.HiddenSize),
            ("ian:max_aspect_len", self.MaxAspectLength),
            ("ian:max_context_len", self.MaxContextLength),
        ]

        return parameters

    # endregion
