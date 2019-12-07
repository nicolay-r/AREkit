import tensorflow as tf
from collections import OrderedDict
from arekit.networks.context.architectures.base import BaseContextNeuralNetwork
from arekit.networks.context.configurations.cnn import CNNConfig
from arekit.networks.tf_helpers import layers


class VanillaCNN(BaseContextNeuralNetwork):
    """
    Title: Relation Classification via Convolutional Deep Neural Network
    Authors: Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao
    Paper: https://www.aclweb.org/anthology/C14-1220/
    Source: https://github.com/roomylee/cnn-relation-extraction

    NOTE: This class is an unofficial implementation of CNN with distance features.
    """

    H_W = "W"
    H_b = "b"
    H_W2 = "W2"
    H_b2 = "b2"
    H_conv_filter = "C"

    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.__hidden = OrderedDict()

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
        result, result_dropout = layers.get_k_layer_pair_logits(g=context_embedding,
                                                                W=W,
                                                                b=b,
                                                                dropout_keep_prob=self.DropoutKeepProb,
                                                                activations=activations)
        return result, result_dropout

    def init_hidden_states(self):
        assert(isinstance(self.Config, CNNConfig))

        self.__hidden[self.H_W] = tf.get_variable(
            name=self.H_W,
            shape=[self.ContextEmbeddingSize, self.Config.HiddenSize],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32)

        self.__hidden[self.H_b] = tf.get_variable(
            name=self.H_b,
            shape=[self.Config.HiddenSize],
            initializer=self.Config.BiasInitializer,
            dtype=tf.float32)

        self.__hidden[self.H_W2] = tf.get_variable(
            name=self.H_W2,
            shape=[self.Config.HiddenSize, self.Config.ClassesCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32)

        self.__hidden[self.H_b2] = tf.get_variable(
            name=self.H_b2,
            shape=[self.Config.ClassesCount],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32)

        self.__hidden[self.H_conv_filter] = tf.get_variable(
            name=self.H_conv_filter,
            shape=[self.Config.WindowSize * self.TermEmbeddingSize, 1, self.Config.FiltersCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32)

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value

    @staticmethod
    def padding(embedded_data, window_size):
        assert(isinstance(window_size, int) and window_size > 0)

        left_padding = (window_size - 1) / 2
        right_padding = (window_size - 1) - left_padding
        return tf.pad(embedded_data, [[0, 0],
                                      [left_padding, right_padding],
                                      [0, 0]])
