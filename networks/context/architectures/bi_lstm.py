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

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.__hidden = {}

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
        self.__hidden[self.H_W] = tf.Variable(
            initial_value=tf.random_normal([self.ContextEmbeddingSize, self.Config.ClassesCount]))
        self.__hidden[self.H_b] = tf.Variable(
            initial_value=tf.random_normal([self.Config.ClassesCount]))

    def iter_hidden_parameters(self):
        for key, value in self.__hidden:
            yield key, value

    @staticmethod
    def __get_cell(hidden_size):
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
