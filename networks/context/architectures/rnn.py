import tensorflow as tf
from core.networks.context.sample import InputSample
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations.rnn import RNNConfig, CellTypes
import utils

# Copyright (c) Joohong Lee
# page: https://github.com/roomylee
# source project: https://github.com/roomylee/rnn-text-classification-tf


class RNN(BaseContextNeuralNetwork):

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(RNN, self).__init__()
        self.__hidden = {}

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, RNNConfig))

        with tf.name_scope("rnn"):
            length = tf.cast(x=utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS)),
                             dtype=tf.int32)
            cell = self.get_cell(self.Config.HiddenSize, self.Config.CellType)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.DropoutKeepProb)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=embedded_terms,
                                               sequence_length=length,
                                               dtype=tf.float32)
            h_outputs = self.__last_relevant(all_outputs, length)

        return h_outputs

    def init_logits_unscaled(self, context_embedding):

        with tf.name_scope("output"):
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(self.__hidden[self.H_W])
            l2_loss += tf.nn.l2_loss(self.__hidden[self.H_b])
            logits = tf.nn.xw_plus_b(context_embedding,
                                     self.__hidden[self.H_W],
                                     self.__hidden[self.H_b],
                                     name="logits")

        return logits, tf.nn.dropout(logits, self.DropoutKeepProb)

    def init_hidden_states(self):
        self.__hidden[self.H_W] = tf.get_variable(shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.__hidden[self.H_b] = tf.Variable(initial_value=tf.constant(0.1, shape=[self.Config.ClassesCount]))

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value

    @staticmethod
    def get_cell(hidden_size, cell_type):
        assert(isinstance(cell_type, unicode))
        if cell_type == CellTypes.RNN:
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == CellTypes.LSTM:
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == CellTypes.GRU:
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            Exception("Incorrect cell_type={}".format(cell_type))
            return None

    @staticmethod
    def __last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)
