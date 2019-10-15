import tensorflow as tf
from collections import OrderedDict
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.sequence import get_cell
from core.networks.context.configurations.rcnn import RCNNConfig
from core.networks.context.sample import InputSample
import utils


class RCNN(BaseContextNeuralNetwork):
    """
    Copyright (c) Joohong Lee
    page: https://github.com/roomylee/rcnn-text-classification
    paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
    """

    H_W = "W"
    H_b = "b"
    H_W2 = "W2"
    H_b2 = "b2"

    def __init__(self):
        super(RCNN, self).__init__()
        self.__hidden = OrderedDict()

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, RCNNConfig))
        text_length = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))

        with tf.name_scope("bi-rnn"):

            fw_cell = get_cell(hidden_size=self.Config.SurroundingOneSideContextEmbeddingSize,
                               cell_type=self.Config.CellType,
                               dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            bw_cell = get_cell(hidden_size=self.Config.SurroundingOneSideContextEmbeddingSize,
                               cell_type=self.Config.CellType,
                               dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
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
            y2 = tf.tanh(tf.einsum('aij,jk->aik', merged, self.__hidden[self.H_W]) + self.__hidden[self.H_b])

        with tf.name_scope("max-pooling"):
            y3 = tf.reduce_max(y2, axis=1)

        return y3

    def init_logits_unscaled(self, context_embedding):
        with tf.name_scope("output"):
            logits = tf.nn.xw_plus_b(context_embedding,
                                     self.__hidden[self.H_W2],
                                     self.__hidden[self.H_b2], name="logits")

        return logits, tf.nn.dropout(logits, self.DropoutKeepProb)

    def init_hidden_states(self):
        assert(isinstance(self.Config, RCNNConfig))

        self.__hidden[self.H_W] = tf.get_variable(
            name=self.H_W,
            shape=[self.__text_embedding_size(), self.Config.HiddenSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.WeightInitializer)

        self.__hidden[self.H_b] = tf.get_variable(
            name=self.H_b,
            shape=[self.Config.HiddenSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

        self.__hidden[self.H_W2] = tf.get_variable(
            name=self.H_W2,
            shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

        self.__hidden[self.H_b2] = tf.get_variable(
            name=self.H_b2,
            shape=[self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value

    def __text_embedding_size(self):
        return self.TermEmbeddingSize + \
               2 * self.Config.SurroundingOneSideContextEmbeddingSize

