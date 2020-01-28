import tensorflow as tf
from collections import OrderedDict

from arekit.networks.context.architectures.base import BaseContextNeuralNetwork
from arekit.networks.context.configurations.rcnn import RCNNConfig
from arekit.networks.context.sample import InputSample
from arekit.networks.data_type import DataType
from arekit.networks.tf_helpers import sequence


class RCNN(BaseContextNeuralNetwork):
    """
    Copyright (c) Joohong Lee
    Title: Recurrent Convolutional Neural Networks for Text Classification
    Paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
    Source: https://github.com/roomylee/rcnn-text-classification
    """

    H_W_text = "W"
    H_b_text = "b"
    H_W2_out = "W2"
    H_b2_out = "b2"

    def __init__(self):
        super(RCNN, self).__init__()
        self.__hidden = OrderedDict()
        self.__dropout_rnn_keep_prob = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    # endregion

    # region public 'set' methods

    def set_input_rnn_keep_prob(self, value):
        self.__dropout_rnn_keep_prob = value

    # endregion

    # region public 'init' methods

    def init_input(self):
        super(RCNN, self).init_input()
        self.__dropout_rnn_keep_prob = tf.placeholder(dtype=tf.float32,
                                                      name="ctx_dropout_rnn_keep_prob")

    def modify_rnn_outputs_optional(self, output_fw, output_bw):
        # Nothing modifies
        return output_fw, output_bw

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, RCNNConfig))
        text_length = sequence.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))

        with tf.name_scope("bi-rnn"):

            fw_cell = sequence.get_cell(hidden_size=self.Config.SurroundingOneSideContextEmbeddingSize,
                                        cell_type=self.Config.CellType,
                                        dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            bw_cell = sequence.get_cell(hidden_size=self.Config.SurroundingOneSideContextEmbeddingSize,
                                        cell_type=self.Config.CellType,
                                        dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            (output_fw, output_bw), states = sequence.bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=embedded_terms,
                sequence_length=text_length,
                dtype=tf.float32)

            output_fw, output_bw = self.modify_rnn_outputs_optional(output_fw, output_bw)

        with tf.name_scope("ctx"):
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            merged = tf.concat([c_left, embedded_terms, c_right], axis=2, name="merged")

        with tf.name_scope("text-representation"):
            y2 = tf.tanh(tf.einsum('aij,jk->aik', merged, self.__hidden[self.H_W_text]) + self.__hidden[self.H_b_text])

        with tf.name_scope("max-pooling"):
            y3 = tf.reduce_max(y2, axis=1)

        return y3

    def init_logits_unscaled(self, context_embedding):
        with tf.name_scope("output"):
            logits = tf.nn.xw_plus_b(context_embedding,
                                     self.__hidden[self.H_W2_out],
                                     self.__hidden[self.H_b2_out], name="logits")

        return logits, tf.nn.dropout(logits, self.DropoutKeepProb)

    def init_body_dependent_hidden_states(self):
        assert(isinstance(self.Config, RCNNConfig))

        self.__hidden[self.H_W_text] = tf.get_variable(
            name=self.H_W_text,
            shape=[self.__text_embedding_size(), self.Config.HiddenSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.WeightInitializer)

        self.__hidden[self.H_b_text] = tf.get_variable(
            name=self.H_b_text,
            shape=[self.Config.HiddenSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

    def init_logits_hidden_states(self):
        assert(isinstance(self.Config, RCNNConfig))

        self.__hidden[self.H_W2_out] = tf.get_variable(
            name=self.H_W2_out,
            shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

        self.__hidden[self.H_b2_out] = tf.get_variable(
            name=self.H_b2_out,
            shape=[self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

    # endregion

    # region public 'iter' methods

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value

    # endregion

    # region public 'create' methods

    def create_feed_dict(self, input, data_type):
        feed_dict = super(RCNN, self).create_feed_dict(input=input, data_type=data_type)
        feed_dict[self.__dropout_rnn_keep_prob] = self.Config.DropoutRNNKeepProb if data_type == DataType.Train else 1.0
        return feed_dict

    # endregion

    # region private methods

    def __text_embedding_size(self):
        return self.TermEmbeddingSize + \
               2 * self.Config.SurroundingOneSideContextEmbeddingSize

    # endregion
