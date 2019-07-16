import tensorflow as tf
from base import DefaultNetworkConfig


class IANConfig(DefaultNetworkConfig):

    __l2_reg = 0.001
    __hidden_size = 128
    __aspect_len = 2
    __aspect_embedding_matrix = None

    def __init__(self):
        super(IANConfig, self).__init__()

    # region Properties

    @property
    def L2Reg(self):
        return self.__l2_reg

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
    def AspectsEmbeddingShape(self):
        return self.__aspect_embedding_matrix.shape

    @property
    def Optimiser(self):
        return tf.train.AdamOptimizer(learning_rate=self.LearningRate)

    # endregion

    def notify_initialization_completed(self):
        self.__aspect_embedding_matrix = self.TermEmbeddingMatrix

    def _internal_get_parameters(self):
        parameters = super(IANConfig, self)._internal_get_parameters()

        parameters += [
            ("ian:l2_reg", self.L2Reg),
            ("ian:hidden_size", self.HiddenSize),
            ("ian:max_aspect_len", self.MaxAspectLength),
            ("ian:max_context_len", self.MaxContextLength),
        ]

        return parameters
