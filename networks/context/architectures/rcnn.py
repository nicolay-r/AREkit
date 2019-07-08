import tensorflow as tf
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.rnn import RNN
from core.networks.context.configurations.rcnn import RCNNConfig
import utils

# Copyright (c) Joohong Lee
# page: https://github.com/roomylee


class RCNN(BaseContextNeuralNetwork):

    def __init__(self):
        super(RCNN, self).__init__()
        # TODO. Use dict for hidden parameters
        self.__W1 = None
        self.__b1 = None
        self.__W2 = None
        self.__b2 = None

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize + \
               self._get_attention_vector_size(self.Config)

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, RCNNConfig))
        text_length = utils.calculate_sequence_length(self.InputX)

        with tf.name_scope("bi-rnn"):
            fw_cell = RNN.get_cell(self.Config.SurroundingOneSideContextEmbeddingSize, self.Config.CellType)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = RNN.get_cell(self.Config.SurroundingOneSideContextEmbeddingSize, self.Config.CellType)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=embedded_terms,
                                                                                       sequence_length=text_length,
                                                                                       dtype=tf.float32)

        with tf.name_scope("ctx"):
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            merged = tf.concat([c_left, embedded_terms, c_right], axis=2, name="merged")

        with tf.name_scope("text-representation"):
            y2 = tf.tanh(tf.einsum('aij,jk->aik', merged, self.__W1) + self.__b1)

        with tf.name_scope("max-pooling"):
            y3 = tf.reduce_max(y2, axis=1)

        if self.Config.UseAttention:
            # TODO. in Nested class
            y3 = tf.concat([y3, self.init_attention_embedding()], axis=-1)

        return y3

    def init_logits_unscaled(self, context_embedding):

        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            l2_loss += tf.nn.l2_loss(self.__W2)
            l2_loss += tf.nn.l2_loss(self.__b2)
            logits = tf.nn.xw_plus_b(context_embedding, self.__W2, self.__b2, name="logits")

        return logits, tf.nn.dropout(logits, self.dropout_keep_prob)

    def init_hidden_states(self):
        assert(isinstance(self.Config, RCNNConfig))

        self.__W1 = tf.Variable(initial_value=tf.random_uniform([self.__text_embedding_size(), self.Config.HiddenSize], -1.0, 1.0),
                                name="W1")
        self.__b1 = tf.Variable(initial_value=tf.constant(0.1, shape=[self.Config.HiddenSize]),
                                name="b1")

        self.__W2 = tf.get_variable(shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    name="W2")
        self.__b2 = tf.get_variable(shape=[self.Config.ClassesCount],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    name="b2")

    # TODO. To Dictionary
    def get_parameters_to_investigate(self):
        return ["W1", "b1", "W2", "b2"], \
               [self.__W1,  self.__b1, self.__W2,  self.__b2]

    def __text_embedding_size(self):
        return self.TermEmbeddingSize + \
               2 * self.Config.SurroundingOneSideContextEmbeddingSize

