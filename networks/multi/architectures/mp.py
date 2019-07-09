import tensorflow as tf
from core.networks.context.architectures.utils import get_two_layer_logits
from core.networks.multi.architectures.base import BaseMultiInstanceNeuralNetwork


class MaxPoolingMultiInstanceNetwork(BaseMultiInstanceNeuralNetwork):
    """
    Provides encoder as a maxpooling over embedded contexts.
    TODO. Provide link
    """
    H_W1 = "W1"
    H_W2 = "W1"
    H_b1 = "b1"
    H_b2 = "b2"

    def __init__(self, context_network):
        super(MaxPoolingMultiInstanceNetwork, self).__init__(context_network)
        self.__hidden = {}

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: [batches, sentences, embedding]
        """
        return self.__contexts_max_pooling(context_outputs=context_outputs,
                                           contexts_per_opinion=self.ContextsPerOpinion)  # [batches, max_pooling]

    def init_hidden_states(self):
        self.__hidden = {
            self.H_W1: tf.Variable(initial_value=tf.random_normal([self.__context_network.ContextEmbeddingSize,
                                                                   self.__cfg.HiddenSize]),
                                   dtype=tf.float32),
            self.H_W2: tf.Variable(initial_value=tf.random_normal([self.__cfg.HiddenSize,
                                                                   self.__cfg.ClassesCount]),
                                   dtype=tf.float32),
            self.H_b1: tf.Variable(initial_value=tf.random_normal([self.__cfg.HiddenSize]),
                                   dtype=tf.float32),
            self.H_b2: tf.Variable(initial_value=tf.random_normal([self.__cfg.ClassesCount]),
                                   dtype=tf.float32)
        }

    def init_logits_unscaled(self, encoded_contexts):
        # TODO. Now it is hardcoded two layer network.
        return get_two_layer_logits(
            encoded_contexts,
            W1=self.__hidden[self.H_W1],
            b1=self.__hidden[self.H_b1],
            W2=self.__hidden[self.H_W2],
            b2=self.__hidden[self.H_b2],
            dropout_keep_prob=self.__dropout_keep_prob,
            activations=[tf.tanh, tf.tanh, None])

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

    def hidden_parameters(self):
        return self.__hidden

