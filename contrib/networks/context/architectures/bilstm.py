"""
Bi-directional Recurrent Neural Network.
Modified version of Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import tensorflow as tf

from arekit.networks.data_type import DataType
from arekit.networks.tf_helpers import sequence
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.networks.context.sample import InputSample
from arekit.contrib.networks.context.architectures.base.fc_single import FullyConnectedLayer


class BiLSTM(FullyConnectedLayer):
    """
    Copyright (c) Joohong Lee
    page: https://github.com/roomylee
    code: https://github.com/roomylee/rnn-text-classification-tf
    """

    def __init__(self):
        super(BiLSTM, self).__init__()
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
        super(BiLSTM, self).init_input()
        self.__dropout_rnn_keep_prob = tf.placeholder(dtype=tf.float32,
                                                      name="ctx_dropout_rnn_keep_prob")

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, BiLSTMConfig))

        with tf.variable_scope("bi-lstm"):

            # Length Calculation
            x_length = sequence.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            # Forward cell
            fw_cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                        cell_type=self.Config.CellType,
                                        lstm_initializer=self.Config.LSTMCellInitializer,
                                        dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            # Backward cell
            bw_cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                        cell_type=self.Config.CellType,
                                        lstm_initializer=self.Config.LSTMCellInitializer,
                                        dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            (output_fw, output_bw), _ = sequence.bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=embedded_terms,
                sequence_length=s_length,
                dtype=tf.float32)

            rnn_outputs = tf.add(output_fw, output_bw)

        return self.customize_rnn_output(rnn_outputs, s_length)

    def customize_rnn_output(self, rnn_outputs, s_length):
        return sequence.select_last_relevant_in_sequence(rnn_outputs, s_length)

    def init_body_dependent_hidden_states(self):
        pass

    # endregion

    # region public 'create' methods

    def create_feed_dict(self, input, data_type):
        feed_dict = super(BiLSTM, self).create_feed_dict(input=input, data_type=data_type)
        feed_dict[self.__dropout_rnn_keep_prob] = self.Config.DropoutRNNKeepProb if data_type == DataType.Train else 1.0
        return feed_dict

    # endregion
