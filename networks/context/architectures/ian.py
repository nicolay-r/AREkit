from collections import OrderedDict

import tensorflow as tf

from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.rnn import RNN
from core.networks.context.architectures.sequence import get_cell
from core.networks.context.configurations.ian import IANConfig
from core.networks.context.configurations.rnn import RNNConfig
from core.networks.context.sample import InputSample
import utils


class IAN(BaseContextNeuralNetwork):
    """
    Paper: https://arxiv.org/pdf/1709.00893.pdf
    Author: Peiqin Lin
    Code: https://github.com/lpq29743/IAN/blob/master/model.py
    """

    H_W = "W"
    H_b = "b"

    def __init__(self):
        super(IAN, self).__init__()
        self.__hidden = OrderedDict()

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, IANConfig))

        with tf.name_scope("rnn"):

            # Length Calculation
            x_length = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            # Forward cell
            cell = get_cell(hidden_size=self.Config.HiddenSize,
                            cell_type=self.Config.CellType,
                            dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

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

#     ASPECT_W = 'W_a'
#     CONTEXT_W = 'W_c'
#     SOFTMAX_W = 'W_l'
#
#     ASPECT_B = 'B_a'
#     CONTEXT_B = 'B_c'
#     SOFTMAX_B = 'B_l'
#
#     def __init__(self):
#         super(IAN, self).__init__()
#         self.__aspects = None
#         self.__E_aspects = None
#         self.__aspect_inputs = None
#
#         self.__aspect_atts = None
#         self.__aspect_reps = None
#         self.__context_atts = None
#         self.__context_reps = None
#
#     @property
#     def ContextEmbeddingSize(self):
#         # return self.Config.HiddenSize * 2
#         return self.Config.HiddenSize
#
#     def init_input(self):
#         super(IAN, self).init_input()
#         assert(isinstance(self.Config, IANConfig))
#         self.__aspects = tf.placeholder(dtype=tf.int32,
#                                         shape=[self.Config.BatchSize, self.Config.MaxAspectLength],
#                                         name="aspects")
#
#         # TODO. Remove it and utilize existed term_embedding instead.
#         self.__E_aspects = tf.get_variable(name="E_aspects",
#                                            dtype=tf.float32,
#                                            initializer=tf.contrib.layers.xavier_initializer(),
#                                            shape=self.Config.AspectsEmbeddingShape,
#                                            trainable=False)
#
#     def init_hidden_states(self):
#         assert(isinstance(self.Config, IANConfig))
#
#         self.w_a = tf.get_variable(
#             name=self.ASPECT_W,
#             shape=[self.Config.HiddenSize, self.Config.HiddenSize],
#             initializer=self.Config.WeightInitializer,
#             regularizer=self.Config.LayerRegularizer,
#             trainable=True)
#
#         self.w_c = tf.get_variable(
#             name=self.CONTEXT_W,
#             shape=[self.Config.HiddenSize, self.Config.HiddenSize],
#             initializer=self.Config.WeightInitializer,
#             regularizer=self.Config.LayerRegularizer,
#             trainable=True)
#
#         self.w_l = tf.get_variable(
#             name=self.SOFTMAX_W,
#             shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
#             initializer=self.Config.WeightInitializer,
#             regularizer=self.Config.LayerRegularizer,
#             trainable=True)
#
#         self.b_a = tf.get_variable(
#             name=self.ASPECT_B,
#             shape=[self.Config.MaxAspectLength, 1],
#             initializer=self.Config.BiasInitializer,
#             regularizer=self.Config.LayerRegularizer,
#             trainable=True)
#
#         self.b_c = tf.get_variable(
#             name=self.CONTEXT_B,
#             shape=[self.Config.MaxContextLength, 1],
#             initializer=self.Config.BiasInitializer,
#             regularizer=self.Config.LayerRegularizer,
#             trainable=True)
#
#         self.b_l = tf.get_variable(
#             name=self.SOFTMAX_B,
#             shape=[self.Config.ClassesCount],
#             initializer=self.Config.BiasInitializer,
#             regularizer=self.Config.LayerRegularizer,
#             trainable=True)
#
#     def init_embedded_input(self):
#
#         context_inputs = super(IAN, self).init_embedded_input()
#
#         # TODO. Fix this, to provide correct embedding.
#         # TODO. Remove e_aspects usage.
#         __aspect_inputs = tf.cast(x=tf.nn.embedding_lookup(params=self.__E_aspects, ids=self.__aspects),
#                                   dtype=tf.float32)
#
#         aspect_inputs = self.process_embedded_data(
#             embedded=__aspect_inputs,
#             dropout_keep_prob=self.EmbeddingDropoutKeepProb)
#
#         return [context_inputs, aspect_inputs]
#
#     def init_context_embedding(self, embedded_terms):
#         assert(isinstance(embedded_terms, list))
#
#         context_inputs, aspect_inputs = embedded_terms
#
#         with tf.name_scope('dynamic_rnn'):
#
#             # Prepare cells
#             # aspect_cell = get_cell(hidden_size=self.Config.HiddenSize,
#             #                        cell_type=self.Config.CellType,
#             #                        dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)
#
#             context_cell = get_cell(hidden_size=self.Config.HiddenSize,
#                                     cell_type=self.Config.CellType,
#                                     dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)
#
#             # Calculate input lengths
#             # aspect_lens = utils.calculate_sequence_length(self.__aspects)
#             # aspect_lens_casted = tf.cast(x=tf.maximum(aspect_lens, 1), dtype=tf.int32)
#
#             context_lens = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
#             context_lens_casted = tf.cast(x=tf.maximum(context_lens, 1), dtype=tf.int32)
#
#             # Receive aspect output
#             # aspect_outputs, _ = tf.nn.dynamic_rnn(cell=aspect_cell,
#             #                                       inputs=aspect_inputs,
#             #                                       sequence_length=aspect_lens_casted,
#             #                                       dtype=tf.float32,
#             #                                       scope='aspect_outputs')
#             # TODO. Select last instead.
#             # TODO. But also keep original (add in config)
#             # aspect_avg = tf.reduce_mean(aspect_outputs, 1)
#             # aspect_avg = utils.select_last_relevant_in_sequence(aspect_outputs, aspect_lens_casted)
#
#             # Receive context output
#             context_outputs, _ = tf.nn.dynamic_rnn(cell=context_cell,
#                                                    inputs=context_inputs,
#                                                    sequence_length=context_lens_casted,
#                                                    dtype=tf.float32,
#                                                    scope='context_outputs')
#             # TODO. Select last instead.
#             # TODO. But also keep original (add in config)
#             # context_avg = tf.reduce_mean(context_outputs, 1)
#             context_avg = utils.select_last_relevant_in_sequence(context_outputs, context_lens_casted)
#
#             # Attention for aspects
#             # aspect_att = tf.nn.softmax(
#             #     tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.__weights['aspect_score'],
#             #                          tf.expand_dims(context_avg, -1)) + self.__biases['aspect_score']))
#             # aspect_rep = tf.reduce_sum(aspect_att * aspect_outputs, 1)
#
            # Attention for context
#            context_att = tf.nn.softmax(
#                tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.__weights['context_score'],
#                                     tf.expand_dims(aspect_avg, -1)) + self.__biases['context_score']))
#            context_rep = tf.reduce_sum(context_att * context_outputs, 1)
#
#            #return tf.concat([context_avg, context_avg], 1)
#            return context_avg
#
#    def init_logits_unscaled(self, context_embedding):
#        return utils.get_k_layer_pair_logits(g=context_embedding,
#                                             W=[self.w_l],
#                                             b=[self.b_l],
#                                             dropout_keep_prob=self.DropoutKeepProb,
#                                             activations=[tf.tanh, None])
#
#    def iter_hidden_parameters(self):
#        yield self.ASPECT_W, self.w_a
#        yield self.CONTEXT_W, self.w_c
#        yield self.SOFTMAX_W, self.w_l
#
#    def create_feed_dict(self, input, data_type):
#        feed_dict = super(IAN, self).create_feed_dict(input, data_type)
        # TODO. Implement a different model with frame_inds.
        # TODO. But original could be based on atitute ends.
#        feed_dict[self.__aspects] = input[InputSample.I_FRAME_INDS]
#        return feed_dict
