import tensorflow as tf
from collections import OrderedDict
from arekit.networks.tf_helpers.layers import get_k_layer_pair_logits
from arekit.contrib.networks.multi.architectures.base import BaseMultiInstanceNeuralNetwork


class BaseMultiInstanceSingleMLP(BaseMultiInstanceNeuralNetwork):
    """
    Single output layer.
    """
    H_W1 = u"W"
    H_b1 = u"b"

    def __init__(self, context_network):
        super(BaseMultiInstanceSingleMLP, self).__init__(context_network)
        self.__hidden = OrderedDict()

    @property
    def EmbeddingSize(self):
        raise NotImplementedError()

    # region 'init' methods

    def init_hidden_states(self):

        self.__hidden[self.H_W1] = tf.get_variable(
            shape=[self.EmbeddingSize, self.Config.ClassesCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32,
            name=self.H_W1)
        self.__hidden[self.H_b1] = tf.get_variable(
            shape=[self.Config.ClassesCount],
            initializer=self.Config.BaseInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32,
            name=self.H_b1)

    def init_logits_unscaled(self, encoded_contexts):
        W = [tensor for var_name, tensor in self.__hidden.iteritems() if 'W' in var_name]
        b = [tensor for var_name, tensor in self.__hidden.iteritems() if 'b' in var_name]
        activations = [tf.tanh] * len(W) + [None]
        return get_k_layer_pair_logits(g=encoded_contexts,
                                       W=W,
                                       b=b,
                                       dropout_keep_prob=self.DropoutKeepProb,
                                       activations=activations)

    def iter_hidden_parameters(self):
        for name, value in super(BaseMultiInstanceSingleMLP, self).iter_hidden_parameters():
            yield name, value

        for name, value in self.__hidden.iteritems():
            yield name, value

    # endregion
