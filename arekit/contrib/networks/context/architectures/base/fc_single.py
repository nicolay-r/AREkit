import tensorflow as tf
from collections import OrderedDict
from arekit.contrib.networks.context.architectures.base.base import SingleInstanceNeuralNetwork


class FullyConnectedLayer(SingleInstanceNeuralNetwork):

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        self.__hidden = OrderedDict()

    def init_logits_unscaled(self, context_embedding):

        with tf.name_scope("output"):
            logits = tf.nn.xw_plus_b(context_embedding,
                                     self.__hidden[self.H_W],
                                     self.__hidden[self.H_b],
                                     name="logits")

        return logits, tf.nn.dropout(logits, self.DropoutKeepProb)

    def init_logits_hidden_states(self):

        self.__hidden[self.H_W] = tf.get_variable(
            name=self.H_W,
            shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer)

        self.__hidden[self.H_b] = tf.get_variable(
            name=self.H_b,
            shape=[self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

    # region public 'iter' methods

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value

    # endregion
