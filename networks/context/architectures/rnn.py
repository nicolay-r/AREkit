import tensorflow as tf
from collections import OrderedDict

from core.networks.context.architectures.sequence import get_cell
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
        self.__hidden = OrderedDict()

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, RNNConfig))

        with tf.name_scope("rnn"):

            # Length Calculation
            x_length = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            # Forward cell
            cell = get_cell(self.Config.HiddenSize, self.Config.CellType)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                                 output_keep_prob=self.DropoutKeepProb)

            # Output
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=embedded_terms,
                                               sequence_length=s_length,
                                               dtype=tf.float32)

            h_outputs = utils.select_last_relevant_in_sequence(all_outputs, s_length)

        return h_outputs

    def init_logits_unscaled(self, context_embedding):

        with tf.name_scope("output"):
            logits = tf.nn.xw_plus_b(context_embedding,
                                     self.__hidden[self.H_W],
                                     self.__hidden[self.H_b],
                                     name="logits")

        return logits, tf.nn.dropout(logits, self.DropoutKeepProb)

    def init_hidden_states(self):

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

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value
