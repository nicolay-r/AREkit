from collections import OrderedDict

import tensorflow as tf
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.configurations.att_bilstm import AttBiLSTMConfig
from core.networks.context.sample import InputSample
import utils


class AttBiLSTM(BaseContextNeuralNetwork):
    """
    Authors: Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
    Paper: https://www.aclweb.org/anthology/P16-2034
    Code author: SeoSangwoo (c), https://github.com/SeoSangwoo
    Code: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    """

    H_W = "W"
    H_b = "b"
    __attention_scope = "attention"

    def __init__(self):
        super(AttBiLSTM, self).__init__()
        self.__att_alphas = None
        self.__hidden = OrderedDict()

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, AttBiLSTMConfig))

        initializer = tf.keras.initializers.glorot_normal

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):

            x_length = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            _fw_cell = tf.nn.rnn_cell.LSTMCell(self.Config.HiddenSize, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.Config.DropoutRNNKeepProb)

            _bw_cell = tf.nn.rnn_cell.LSTMCell(self.Config.HiddenSize, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.Config.DropoutRNNKeepProb)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                             cell_bw=bw_cell,
                                                             inputs=embedded_terms,
                                                             sequence_length=s_length,
                                                             dtype=tf.float32)

            rnn_outputs = tf.add(rnn_outputs[0], rnn_outputs[1])

        # Attention
        with tf.variable_scope('attention'):
            attn, self.__att_alphas = self.__attention(rnn_outputs)

        return attn

    @staticmethod
    def __attention(inputs):
        # Trainable parameters
        hidden_size = inputs.shape[2].value
        u_omega = tf.get_variable(name="u_omega",
                                  shape=[hidden_size],
                                  initializer=tf.keras.initializers.glorot_normal())

        with tf.name_scope('v'):
            v = tf.tanh(inputs)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        # Final output with tanh
        output = tf.tanh(output)

        return output, alphas

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
        for key, value in self.__hidden.iteritems():
            yield key, value

        yield "ATT_Weights", self.__att_alphas
