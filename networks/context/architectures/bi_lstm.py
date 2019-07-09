"""
Bi-directional Recurrent Neural Network.
Modified version of Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import utils
import tensorflow as tf
from tensorflow.contrib import rnn
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations.bi_lstm import BiLSTMConfig


class BiLSTM(BaseContextNeuralNetwork):

    def __init__(self):
        super(BiLSTM, self).__init__()
        # TODO: Use dict for hidden paramters
        self.__W = None
        self.__b = None

    @property
    def ContextEmbeddingSize(self):
        return 2 * self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, BiLSTMConfig))

        x = tf.unstack(embedded_terms, axis=1)
        lstm_fw_cell = BiLSTM.__get_cell(self.Config.HiddenSize)
        lstm_bw_cell = BiLSTM.__get_cell(self.Config.HiddenSize)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                     output_keep_prob=self.DropoutKeepProb)

        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                     lstm_bw_cell,
                                                     x,
                                                     dtype=tf.float32)
        return outputs[-1]

    def init_logits_unscaled(self, context_embedding):
        return utils.get_k_layer_pair_logits(
            g=context_embedding,
            W=[self.__W],
            b=[self.__b],
            dropout_keep_prob=self.DropoutKeepProb,
            activations=[tf.tanh, None])

    def init_hidden_states(self):
        self.__W = tf.Variable(initial_value=tf.random_normal([self.ContextEmbeddingSize, self.Config.ClassesCount]),
                               name="W")
        self.__b = tf.Variable(initial_value=tf.random_normal([self.Config.ClassesCount]),
                               name="b")

    # TODO. To Dictionary
    def hidden_parameters(self):
        return ["W", "b"], \
               [self.__W, self.__b]

    @staticmethod
    def __get_cell(hidden_size):
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
