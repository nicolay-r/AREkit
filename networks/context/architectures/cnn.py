import tensorflow as tf
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations.cnn import CNNConfig
import utils


class VanillaCNN(BaseContextNeuralNetwork):

    H_W = "W"
    H_b = "b"
    H_W2 = "W2"
    H_b2 = "b2"
    H_conv_filter = "C"

    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.__hidden = {}

    @property
    def Hidden(self):
        return self.__hidden

    @property
    def ContextEmbeddingSize(self):
        return self.Config.FiltersCount

    def init_context_embedding(self, embedded_terms):
        embedding = self.init_context_embedding_core(embedded_terms)
        return tf.concat(embedding, axis=-1)

    def init_context_embedding_core(self, embedded_terms):
        embedded_terms = self.padding(embedded_terms, self.Config.WindowSize)

        bwc_line = tf.reshape(embedded_terms,
                              [self.Config.BatchSize,
                               (self.Config.TermsPerContext + (self.Config.WindowSize - 1)) * self.TermEmbeddingSize,
                               1])

        bwc_conv = tf.nn.conv1d(bwc_line, self.__hidden[self.H_conv_filter], self.TermEmbeddingSize,
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

        return g

    def init_logits_unscaled(self, context_embedding):
        W = [tensor for var_name, tensor in self.__hidden.iteritems() if 'W' in var_name]
        b = [tensor for var_name, tensor in self.__hidden.iteritems() if 'b' in var_name]
        activations = [tf.tanh] * len(W)
        activations.append(None)
        return utils.get_k_layer_pair_logits(g=context_embedding,
                                             W=W,
                                             b=b,
                                             dropout_keep_prob=self.DropoutKeepProb,
                                             activations=activations)

    def init_hidden_states(self):
        assert(isinstance(self.Config, CNNConfig))
        self.__hidden = {
            self.H_W: tf.Variable(initial_value=tf.random_normal([self.ContextEmbeddingSize,
                                                                  self.Config.HiddenSize]),
                                  dtype=tf.float32),
            self.H_b: tf.Variable(initial_value=tf.random_normal([self.Config.HiddenSize]),
                                  dtype=tf.float32),
            self.H_W2: tf.Variable(initial_value=tf.random_normal([self.Config.HiddenSize,
                                                                   self.Config.ClassesCount]),
                                   dtype=tf.float32),
            self.H_b2: tf.Variable(initial_value=tf.random_normal([self.Config.ClassesCount]),
                                   dtype=tf.float32),
            self.H_conv_filter: tf.Variable(
                initial_value=tf.random_normal([self.Config.WindowSize * self.TermEmbeddingSize,
                                                1,
                                                self.Config.FiltersCount]),
                dtype=tf.float32)
        }

    def iter_hidden_parameters(self):
        for key, value in self.__hidden:
            yield key, value

    @staticmethod
    def padding(embedded_data, window_size):
        assert(isinstance(window_size, int) and window_size > 0)

        left_padding = (window_size - 1) / 2
        right_padding = (window_size - 1) - left_padding
        return tf.pad(embedded_data, [[0, 0],
                                      [left_padding, right_padding],
                                      [0, 0]])
