import tensorflow as tf
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations.cnn import CNNConfig
import utils


class VanillaCNN(BaseContextNeuralNetwork):

    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.W = None
        self.b = None
        self.W2 = None
        self.b2 = None
        self.conv_filter = None

    @property
    def ContextEmbeddingSize(self):
        return self.Config.FiltersCount + \
               self._get_attention_vector_size(self.Config)

    def init_context_embedding(self, embedded_terms):
        embedded_terms = self.padding(embedded_terms, self.Config.WindowSize)

        bwc_line = tf.reshape(embedded_terms,
                              [self.Config.BatchSize,
                               (self.Config.TermsPerContext + (self.Config.WindowSize - 1)) * self.TermEmbeddingSize,
                               1])

        bwc_conv = tf.nn.conv1d(bwc_line, self.conv_filter, self.TermEmbeddingSize,
                                "VALID",
                                data_format="NHWC",
                                name="conv")

        bwgc_conv = tf.reshape(bwc_conv, [self.Config.BatchSize,
                                          1,
                                          self.Config.TermsPerContext,
                                          self.Config.FiltersCount])

        # Max Pooling
        bwgc_mpool = tf.nn.max_pool(
                bwgc_conv,
                [1, 1, self.Config.TermsPerContext, 1],
                [1, 1, self.Config.TermsPerContext, 1],
                padding='VALID',
                data_format="NHWC")

        bc_mpool = tf.squeeze(bwgc_mpool, axis=[1, 2])

        g = tf.reshape(bc_mpool, [self.Config.BatchSize, self.Config.FiltersCount])

        if self.Config.UseAttention:
            g = tf.concat([g, self.init_attention_embedding()], axis=-1)

        return tf.concat(g, axis=-1)

    def init_logits_unscaled(self, context_embedding):
        return utils.get_two_layer_logits(
            g=context_embedding,
            W1=self.W,
            b1=self.b,
            W2=self.W2,
            b2=self.b2,
            dropout_keep_prob=self.dropout_keep_prob,
            activations=[tf.tanh, tf.tanh, None])

    def init_hidden_states(self):
        assert(isinstance(self.Config, CNNConfig))
        self.W = tf.Variable(initial_value=tf.random_normal([self.ContextEmbeddingSize, self.Config.HiddenSize]),
                             dtype=tf.float32,
                             name="W")
        self.b = tf.Variable(initial_value=tf.random_normal([self.Config.HiddenSize]),
                             dtype=tf.float32,
                             name="b")
        self.W2 = tf.Variable(initial_value=tf.random_normal([self.Config.HiddenSize, self.Config.ClassesCount]),
                              dtype=tf.float32,
                              name="W2")
        self.b2 = tf.Variable(initial_value=tf.random_normal([self.Config.ClassesCount]),
                              dtype=tf.float32,
                              name="b2")
        self.conv_filter = tf.Variable(initial_value=tf.random_normal([self.Config.WindowSize * self.TermEmbeddingSize, 1, self.Config.FiltersCount]),
                                       dtype=tf.float32,
                                       name="C")

    # TODO. To Dictionary
    def get_parameters_to_investigate(self):
        return ["W", "b", "W2", "b2", "C"], \
               [self.W, self.b, self.W2, self.b2, self.conv_filter]

    @staticmethod
    def padding(embedded_data, window_size):
        assert(isinstance(window_size, int) and window_size > 0)

        left_padding = (window_size - 1) / 2
        right_padding = (window_size - 1) - left_padding
        return tf.pad(embedded_data, [[0, 0],
                                      [left_padding, right_padding],
                                      [0, 0]])
