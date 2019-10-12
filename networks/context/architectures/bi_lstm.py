"""
Bi-directional Recurrent Neural Network.
Modified version of Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import utils
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import OrderedDict
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.sequence import get_cell
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.sample import InputSample


class BiLSTM(BaseContextNeuralNetwork):

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.__hidden = OrderedDict()

    @property
    def ContextEmbeddingSize(self):
        return 2 * self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, BiLSTMConfig))

        with tf.name_scope("bi-lstm"):
            x = tf.unstack(embedded_terms, axis=1)

            # Length Calculation
            x_length = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            # Forward cell
            fw_cell = get_cell(hidden_size=self.Config.HiddenSize,
                               cell_type=self.Config.CellType,
                               dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            # Backward cell
            bw_cell = get_cell(hidden_size=self.Config.HiddenSize,
                               cell_type=self.Config.CellType,
                               dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            h_output_list, _, _ = rnn.static_bidirectional_rnn(cell_fw=fw_cell,
                                                               cell_bw=bw_cell,
                                                               inputs=x,
                                                               sequence_length=s_length,
                                                               dtype=tf.float32)
            # [terms_per_ctx, batch, emb_size]
            h_output_tensor = tf.convert_to_tensor(h_output_list, dtype=tf.float32)
            # [batch, terms_per_ctx, emb_size]
            h_output_tensor = tf.transpose(h_output_tensor, perm=[1, 0, 2])

        return utils.select_last_relevant_in_sequence(h_output_tensor, s_length)

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
