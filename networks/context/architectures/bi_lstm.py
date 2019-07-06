"""
Bi-directional Recurrent Neural Network.
Modified version of Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import tensorflow as tf
from tensorflow.contrib import rnn

from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations import BiLSTMConfig

import utils


class BiLSTM(BaseContextNeuralNetwork):

    def __init__(self):
        super(BiLSTM, self).__init__()
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
                                                     output_keep_prob=self.dropout_keep_prob)

        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                     lstm_bw_cell,
                                                     x,
                                                     dtype=tf.float32)
        return outputs[-1]

    def init_logits_unscaled(self, context_embedding):
        return utils.get_single_layer_logits(context_embedding, self.__W, self.__b, self.dropout_keep_prob)

    def init_hidden_states(self):
        self.__W = tf.Variable(tf.random_normal([self.ContextEmbeddingSize, self.Config.ClassesCount]))
        self.__b = tf.Variable(tf.random_normal([self.Config.ClassesCount]))

    @staticmethod
    def __get_cell(hidden_size):
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
