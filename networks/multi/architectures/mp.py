import tensorflow as tf
from collections import OrderedDict
from core.networks.tf_helpers.layers import get_k_layer_pair_logits
from core.networks.multi.architectures.base import BaseMultiInstanceNeuralNetwork


class MaxPoolingMultiInstanceNetwork(BaseMultiInstanceNeuralNetwork):
    """
    Paper: https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf
    Authors: Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang
    """
    H_W1 = "W"
    H_W2 = "W2"
    H_b1 = "b"
    H_b2 = "b2"

    def __init__(self, context_network):
        super(MaxPoolingMultiInstanceNetwork, self).__init__(context_network)
        self.__hidden = OrderedDict()

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: [batches, sentences, embedding]
        """
        return self.__contexts_max_pooling(context_outputs=context_outputs,
                                           contexts_per_opinion=self.ContextsPerOpinion)  # [batches, max_pooling]

    def init_hidden_states(self):
        self.__hidden[self.H_W1] = tf.get_variable(
            shape=[self.ContextNetwork.ContextEmbeddingSize, self.Config.HiddenSize],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32,
            name=self.H_W1)
        self.__hidden[self.H_W2] = tf.get_variable(
            shape=[self.Config.HiddenSize, self.Config.ClassesCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32,
            name=self.H_W2)
        self.__hidden[self.H_b1] = tf.get_variable(
            shape=[self.Config.HiddenSize],
            initializer=self.Config.BaseInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32,
            name=self.H_b1)
        self.__hidden[self.H_b2] = tf.get_variable(
            shape=[self.Config.ClassesCount],
            initializer=self.Config.BaseInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32,
            name=self.H_b2)

    def init_logits_unscaled(self, encoded_contexts):
        W = [tensor for var_name, tensor in self.__hidden.iteritems() if 'W' in var_name]
        b = [tensor for var_name, tensor in self.__hidden.iteritems() if 'b' in var_name]
        activations = [tf.tanh] * len(W) + [None]
        return get_k_layer_pair_logits(g=encoded_contexts,
                                       W=W,
                                       b=b,
                                       dropout_keep_prob=self.DropoutKeepProb,
                                       activations=activations)

    @staticmethod
    def __contexts_max_pooling(context_outputs, contexts_per_opinion):
        context_outputs = tf.expand_dims(context_outputs, 0)     # [1, batches, sentences, embedding]
        context_outputs = tf.nn.max_pool(
            context_outputs,
            ksize=[1, 1, contexts_per_opinion, 1],
            strides=[1, 1, contexts_per_opinion, 1],
            padding='VALID',
            data_format="NHWC")
        return tf.squeeze(context_outputs)                       # [batches, max_pooling]

    def iter_hidden_parameters(self):
        for name, value in self.__hidden.iteritems():
            yield name, value

