import tensorflow as tf

from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.sequence import get_cell
from core.networks.context.configurations.ian import IANConfig
from core.networks.context.sample import InputSample
import utils


class IAN(BaseContextNeuralNetwork):
    """
    Paper: https://arxiv.org/pdf/1709.00893.pdf
    Author: Peiqin Lin
    Code: https://github.com/lpq29743/IAN/blob/master/model.py
    """

    def __init__(self):
        super(IAN, self).__init__()
        self.__aspects = None
        self.__E_aspects = None
        # TODO. May use {} instead on None at init.
        self.__weights = None
        # TODO. May use {} instead on None at init.
        self.__biases = None
        self.__aspect_inputs = None

        self.__aspect_atts = None
        self.__aspect_reps = None
        self.__context_atts = None
        self.__context_reps = None

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize * 2

    def init_input(self):
        super(IAN, self).init_input()
        assert(isinstance(self.Config, IANConfig))
        self.__aspects = tf.placeholder(dtype=tf.int32,
                                        shape=[self.Config.BatchSize, self.Config.MaxAspectLength],
                                        name="aspects")

        # TODO. Remove it and utilize existed term_embedding instead.
        self.__E_aspects = tf.get_variable(name="E_aspects",
                                           dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           shape=self.Config.AspectsEmbeddingShape,
                                           trainable=False)

    def init_hidden_states(self):
        assert(isinstance(self.Config, IANConfig))

        with tf.name_scope('weights'):
            self.__weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.Config.HiddenSize, self.Config.HiddenSize],
                    initializer=self.Config.WeightInitializer,
                    regularizer=self.Config.LayerRegularizer,
                    trainable=True
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.Config.HiddenSize, self.Config.HiddenSize],
                    initializer=self.Config.WeightInitializer,
                    regularizer=self.Config.LayerRegularizer,
                    trainable=True
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
                    initializer=self.Config.WeightInitializer,
                    regularizer=self.Config.LayerRegularizer,
                    trainable=True
                ),
            }

        with tf.name_scope('biases'):
            self.__biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.Config.MaxAspectLength, 1],
                    initializer=self.Config.BiasInitializer,
                    regularizer=self.Config.LayerRegularizer,
                    trainable=True
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.Config.MaxContextLength, 1],
                    initializer=self.Config.BiasInitializer,
                    regularizer=self.Config.LayerRegularizer,
                    trainable=True
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.Config.ClassesCount],
                    initializer=self.Config.BiasInitializer,
                    regularizer=self.Config.LayerRegularizer,
                    trainable=True
                ),
            }

    def init_embedded_input(self):

        context_inputs = super(IAN, self).init_embedded_input()

        # TODO. Fix this, to provide correct embedding.
        # TODO. Remove e_aspects usage.
        __aspect_inputs = tf.cast(x=tf.nn.embedding_lookup(params=self.__E_aspects, ids=self.__aspects),
                                  dtype=tf.float32)

        aspect_inputs = self.process_embedded_data(
            embedded=__aspect_inputs,
            dropout_keep_prob=self.EmbeddingDropoutKeepProb)

        return [context_inputs, aspect_inputs]

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(embedded_terms, list))

        context_inputs, aspect_inputs = embedded_terms

        with tf.name_scope('dynamic_rnn'):

            # Prepare cells
            aspect_cell = get_cell(hidden_size=self.Config.HiddenSize,
                                   cell_type=self.Config.CellType,
                                   dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            context_cell = get_cell(hidden_size=self.Config.HiddenSize,
                                    cell_type=self.Config.CellType,
                                    dropout_rnn_keep_prob=self.Config.DropoutRNNKeepProb)

            # Calculate input lengths
            aspect_lens = utils.calculate_sequence_length(self.__aspects)
            aspect_lens_casted = tf.cast(x=tf.maximum(aspect_lens, 1), dtype=tf.int32)

            context_lens = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            context_lens_casted = tf.cast(x=tf.maximum(context_lens, 1), dtype=tf.int32)

            # Receive aspect output
            aspect_outputs, _ = tf.nn.dynamic_rnn(cell=aspect_cell,
                                                  inputs=aspect_inputs,
                                                  sequence_length=aspect_lens_casted,
                                                  dtype=tf.float32,
                                                  scope='aspect_outputs')
            # TODO. Select last instead.
            # TODO. But also keep original (add in config)
            # aspect_avg = tf.reduce_mean(aspect_outputs, 1)
            aspect_avg = utils.select_last_relevant_in_sequence(aspect_outputs, aspect_lens_casted)

            # Receive context output
            context_outputs, _ = tf.nn.dynamic_rnn(cell=context_cell,
                                                   inputs=context_inputs,
                                                   sequence_length=context_lens_casted,
                                                   dtype=tf.float32,
                                                   scope='context_outputs')
            # TODO. Select last instead.
            # TODO. But also keep original (add in config)
            # context_avg = tf.reduce_mean(context_outputs, 1)
            context_avg = utils.select_last_relevant_in_sequence(context_outputs, context_lens_casted)

            # Attention for aspects
            # aspect_att = tf.nn.softmax(
            #     tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.__weights['aspect_score'],
            #                          tf.expand_dims(context_avg, -1)) + self.__biases['aspect_score']))
            # aspect_rep = tf.reduce_sum(aspect_att * aspect_outputs, 1)

            # # Attention for context
            # context_att = tf.nn.softmax(
            #     tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.__weights['context_score'],
            #                          tf.expand_dims(aspect_avg, -1)) + self.__biases['context_score']))
            # context_rep = tf.reduce_sum(context_att * context_outputs, 1)

            return tf.concat([aspect_avg, context_avg], 1)

    def init_logits_unscaled(self, context_embedding):
        return utils.get_k_layer_pair_logits(g=context_embedding,
                                             W=[self.__weights['softmax']],
                                             b=[self.__biases['softmax']],
                                             dropout_keep_prob=self.DropoutKeepProb,
                                             activations=[tf.tanh, None])

    def iter_hidden_parameters(self):
        assert(isinstance(self.__weights, dict))
        assert(isinstance(self.__biases, dict))

        for key, value in self.__weights.iteritems():
            yield u'w_{}'.format(key), value

        for key, value in self.__biases.iteritems():
            yield u'b_{}'.format(key), value

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IAN, self).create_feed_dict(input, data_type)
        # TODO. Implement a different model with frame_inds.
        # TODO. But original could be based on atitute ends.
        feed_dict[self.__aspects] = input[InputSample.I_FRAME_INDS]
        return feed_dict
