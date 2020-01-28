import tensorflow as tf
from collections import OrderedDict

from arekit.networks.data_type import DataType
from arekit.networks.tf_helpers import sequence
from arekit.networks.context.sample import InputSample
from arekit.networks.context.architectures.base import BaseContextNeuralNetwork
from arekit.networks.context.configurations.rnn import RNNConfig


class RNN(BaseContextNeuralNetwork):
    """
    Copyright (c) Joohong Lee
    page: https://github.com/roomylee
    code: https://github.com/roomylee/rnn-text-classification-tf
    """

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(RNN, self).__init__()
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
        super(RNN, self).init_input()
        self.__dropout_rnn_keep_prob = tf.placeholder(dtype=tf.float32,
                                                      name="ctx_dropout_rnn_keep_prob")

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, RNNConfig))

        with tf.name_scope("rnn"):

            # Length Calculation
            x_length = sequence.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            # Forward cell
            cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                     cell_type=self.Config.CellType,
                                     dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            # Output
            all_outputs, _ = sequence.rnn(cell=cell,
                                          inputs=embedded_terms,
                                          sequence_length=s_length,
                                          dtype=tf.float32)

            h_outputs = sequence.select_last_relevant_in_sequence(all_outputs, s_length)

        return h_outputs

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

    def init_body_dependent_hidden_states(self):
        pass

    # endregion

    # region public 'create' methods

    def create_feed_dict(self, input, data_type):
        feed_dict = super(RNN, self).create_feed_dict(input=input, data_type=data_type)
        feed_dict[self.__dropout_rnn_keep_prob] = self.Config.DropoutRNNKeepProb if data_type == DataType.Train else 1.0
        return feed_dict

    # endregion

    # region public 'iter' methods

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value

    # endregion

