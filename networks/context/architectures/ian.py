import tensorflow as tf

from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.sequence import get_cell
from core.networks.context.configurations.ian import IANConfig, StatesAggregationModes
from core.networks.context.sample import InputSample
import utils


class IAN(BaseContextNeuralNetwork):
    """
    Paper: https://arxiv.org/pdf/1709.00893.pdf
    Author: Peiqin Lin
    Code: https://github.com/lpq29743/IAN/blob/master/model.py
    """
    ASPECT_W = 'W_a'
    CONTEXT_W = 'W_c'
    SOFTMAX_W = 'W_l'

    ASPECT_B = 'B_a'
    CONTEXT_B = 'B_c'
    SOFTMAX_B = 'B_l'

    def __init__(self):
        super(IAN, self).__init__()
        self.__aspects = None

        # Hidden states
        self.__w_a = None
        self.__w_c = None
        self.__w_l = None
        self.__b_a = None
        self.__b_c = None
        self.__b_l = None

        # Input dependent parameters
        self.__aspect_att = None
        self.__context_att = None

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize * 2

    def init_input(self):
        super(IAN, self).init_input()
        assert(isinstance(self.Config, IANConfig))
        self.__aspects = tf.placeholder(dtype=tf.int32,
                                        shape=[self.Config.BatchSize, self.Config.MaxAspectLength],
                                        name="aspects")

    def init_hidden_states(self):
        assert(isinstance(self.Config, IANConfig))

        self.__w_a = tf.get_variable(
            name=self.ASPECT_W,
            shape=[self.Config.HiddenSize, self.Config.HiddenSize],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__w_c = tf.get_variable(
            name=self.CONTEXT_W,
            shape=[self.Config.HiddenSize, self.Config.HiddenSize],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__w_l = tf.get_variable(
            name=self.SOFTMAX_W,
            shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__b_a = tf.get_variable(
            name=self.ASPECT_B,
            shape=[self.Config.MaxAspectLength, 1],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__b_c = tf.get_variable(
            name=self.CONTEXT_B,
            shape=[self.Config.MaxContextLength, 1],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__b_l = tf.get_variable(
            name=self.SOFTMAX_B,
            shape=[self.Config.ClassesCount],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

    def init_embedded_input(self):

         context_embedded = super(IAN, self).init_embedded_input()

         # TODO. Provide pos_embedding.
         # TODO. Provide dist_embedding.
         aspect_inputs = tf.cast(
             x=tf.nn.embedding_lookup(params=self.TermEmbedding, ids=self.__aspects),
             dtype=tf.float32)

         aspect_embedded = self.process_embedded_data(
             embedded=aspect_inputs,
             dropout_keep_prob=self.EmbeddingDropoutKeepProb)

         return [context_embedded, aspect_embedded]

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
            aspect_avg = self.__aggreagate(self.Config,
                                           outputs=aspect_outputs,
                                           length=aspect_lens_casted)

            # Receive context output
            context_outputs, _ = tf.nn.dynamic_rnn(cell=context_cell,
                                                   inputs=context_inputs,
                                                   sequence_length=context_lens_casted,
                                                   dtype=tf.float32,
                                                   scope='context_outputs')
            context_avg = self.__aggreagate(self.Config,
                                            outputs=context_outputs,
                                            length=context_lens_casted)

            # Attention for aspects
            self.__aspect_att = tf.nn.softmax(
                tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.__w_a,
                                     tf.expand_dims(context_avg, -1)) + self.__b_a))
            aspect_rep = tf.reduce_sum(self.__aspect_att * aspect_outputs, 1)

            # Attention for context
            self.__context_att = tf.nn.softmax(
                tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.__w_c,
                                     tf.expand_dims(aspect_avg, -1)) + self.__b_c))
            context_rep = tf.reduce_sum(self.__context_att * context_outputs, 1)

            return tf.concat([context_rep, aspect_rep], 1)

    @staticmethod
    def __aggreagate(config, outputs, length):
        assert(isinstance(config, IANConfig))
        if config.StatesAggregationMode == StatesAggregationModes.AVERAGE:
            return tf.reduce_mean(outputs, 1)
        if config.StatesAggregationMode == StatesAggregationModes.LAST_IN_SEQUENCE:
            return utils.select_last_relevant_in_sequence(outputs, length)
        else:
            raise Exception('"{}" type does not supported'.format(config.StatesAggregationMode))

    def init_logits_unscaled(self, context_embedding):
        return utils.get_k_layer_pair_logits(g=context_embedding,
                                             W=[self.__w_l],
                                             b=[self.__b_l],
                                             dropout_keep_prob=self.DropoutKeepProb,
                                             activations=[tf.tanh, None])

    def iter_hidden_parameters(self):
        yield self.ASPECT_W, self.__w_a
        yield self.CONTEXT_W, self.__w_c
        yield self.SOFTMAX_W, self.__w_l
        yield self.ASPECT_B, self.__b_a
        yield self.CONTEXT_B, self.__b_c
        yield self.SOFTMAX_B, self.__b_l

    def iter_input_dependent_hidden_parameters(self):
        yield u'aspect_att', self.__aspect_att
        yield u'context_att', self.__context_att

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IAN, self).create_feed_dict(input, data_type)

        feed_dict[self.__aspects] = input[InputSample.I_FRAME_INDS]

        return feed_dict
