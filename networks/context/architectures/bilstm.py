"""
Bi-directional Recurrent Neural Network.
Modified version of Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from collections import OrderedDict
import tensorflow as tf
from core.networks.tf_helpers import layers
from core.networks.tf_helpers import sequence
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.sample import InputSample


class BiLSTM(BaseContextNeuralNetwork):
    """
    Copyright (c) Joohong Lee
    page: https://github.com/roomylee
    code: https://github.com/roomylee/rnn-text-classification-tf
    """

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.__hidden = OrderedDict()

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, BiLSTMConfig))

        with tf.name_scope("bi-lstm"):

            # Length Calculation
            x_length = sequence.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            # Forward cell
            fw_cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                        cell_type=self.Config.CellType,
                                        dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            # Backward cell
            bw_cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                        cell_type=self.Config.CellType,
                                        dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            (output_fw, output_bw), _ = sequence.bidirectional_rnn(cell_fw=fw_cell,
                                                                   cell_bw=bw_cell,
                                                                   inputs=embedded_terms,
                                                                   sequence_length=s_length,
                                                                   dtype=tf.float32)

            rnn_outputs = tf.add(output_fw, output_bw)

        return sequence.select_last_relevant_in_sequence(rnn_outputs, s_length)

    def init_logits_unscaled(self, context_embedding):
        W = [tensor for var_name, tensor in self.__hidden.iteritems() if 'W' in var_name]
        b = [tensor for var_name, tensor in self.__hidden.iteritems() if 'b' in var_name]
        activations = [tf.tanh] * len(W)
        activations.append(None)
        return layers.get_k_layer_pair_logits(g=context_embedding,
                                              W=W,
                                              b=b,
                                              dropout_keep_prob=self.DropoutKeepProb,
                                              activations=activations)

    def init_hidden_states(self):
        self.__hidden[self.H_W] = tf.get_variable(
            name=self.H_W,
            shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32)

        self.__hidden[self.H_b] = tf.get_variable(
            name=self.H_b,
            shape=[self.Config.ClassesCount],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            dtype=tf.float32)

    def iter_hidden_parameters(self):
        for key, value in self.__hidden.iteritems():
            yield key, value
