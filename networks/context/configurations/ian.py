import tensorflow as tf
from base import DefaultNetworkConfig


class IANConfig(DefaultNetworkConfig):

    __l2_reg = 0.001
    __hidden_size = 128
    __aspect_len = 2
    __aspect_embedding_matrix = None

    def __init__(self):
        super(IANConfig, self).__init__()

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

    def set_term_embedding(self, embedding_matrix):
        super(IANConfig, self).set_term_embedding(embedding_matrix)
        self.__aspect_embedding_matrix = embedding_matrix

    @property
    def Optimiser(self):
        return tf.train.AdamOptimizer(learning_rate=self.LearningRate)

    def _internal_get_parameters(self):
        parameters = super(IANConfig, self)._internal_get_parameters()

        parameters += [
            ("ian:l2_reg", self.L2Reg),
            ("ian:hidden_size", self.HiddenSize),
            ("ian:max_aspect_len", self.MaxAspectLength),
            ("ian:max_context_len", self.MaxContextLength),
        ]

        return parameters
