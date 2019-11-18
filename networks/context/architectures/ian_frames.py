import tensorflow as tf

import core.networks.tf_helpers.initialization
import core.networks.tf_helpers.sequence
from core.networks.attention.helpers import embedding
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.data_type import DataType
from core.networks.tf_helpers.sequence import get_cell
from core.networks.context.configurations.ian_frames import IANFramesConfig, StatesAggregationModes
from core.networks.context.sample import InputSample
from core.networks.tf_helpers import layers, sequence
from core.networks.tf_helpers.filtering import filter_batch_elements, select_entity_related_elements


class IANFrames(BaseContextNeuralNetwork):
    """
    Title: Interactive Attention Networks for Aspect-Level Sentiment Classification
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
        super(IANFrames, self).__init__()

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

        self.__dropout_rnn_keep_prob = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize * 2

    @property
    def AspectInput(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)

    # endregion

    # region public 'init' methods

    def init_input(self):
        super(IANFrames, self).init_input()
        self.__dropout_rnn_keep_prob = tf.placeholder(dtype=tf.float32,
                                                      name="ctx_dropout_rnn_keep_prob")

    def init_hidden_states(self):
        assert(isinstance(self.Config, IANFramesConfig))

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

         context_embedded = super(IANFrames, self).init_embedded_input()
         aspect_embedded = self.__compose_all_parameters()

         return [context_embedded, aspect_embedded]

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(embedded_terms, list))

        context_embedded, aspects_embedded = embedded_terms

        with tf.name_scope('dynamic_rnn'):

            # Prepare cells
            aspect_cell = get_cell(hidden_size=self.Config.HiddenSize,
                                   cell_type=self.Config.CellType,
                                   dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            context_cell = get_cell(hidden_size=self.Config.HiddenSize,
                                    cell_type=self.Config.CellType,
                                    dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            # Calculate input lengths
            aspect_lens = core.networks.tf_helpers.sequence.calculate_sequence_length(
                sequence=self.AspectInput,
                is_neg_placeholder=InputSample.FRAMES_PAD_VALUE < 0)

            aspect_lens_casted = tf.cast(x=tf.maximum(aspect_lens, 1), dtype=tf.int32)

            context_lens = core.networks.tf_helpers.sequence.calculate_sequence_length(
                sequence=self.get_input_parameter(InputSample.I_X_INDS))

            context_lens_casted = tf.cast(x=tf.maximum(context_lens, 1), dtype=tf.int32)

            # Receive aspect output
            aspect_outputs, _ = sequence.rnn(cell=aspect_cell,
                                             inputs=aspects_embedded,
                                             sequence_length=aspect_lens_casted,
                                             dtype=tf.float32,
                                             scope='aspect_outputs')
            aspect_avg = self.__aggreagate(self.Config,
                                           outputs=aspect_outputs,
                                           length=aspect_lens_casted)

            # Receive context output
            context_outputs, _ = sequence.rnn(cell=context_cell,
                                              inputs=context_embedded,
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
            aspect_rep = tf.reduce_sum(self.__aspect_att * aspect_outputs, axis=1)

            # Attention for context
            self.__context_att = tf.nn.softmax(
                tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.__w_c,
                                     tf.expand_dims(aspect_avg, -1)) + self.__b_c))
            context_rep = tf.reduce_sum(self.__context_att * context_outputs, axis=1)

            return tf.concat([context_rep, aspect_rep], 1)

    def init_logits_unscaled(self, context_embedding):
        return layers.get_k_layer_pair_logits(g=context_embedding,
                                              W=[self.__w_l],
                                              b=[self.__b_l],
                                              dropout_keep_prob=self.DropoutKeepProb)

    # endregion

    # region public 'iter' methods

    def iter_hidden_parameters(self):
        yield self.ASPECT_W, self.__w_a
        yield self.CONTEXT_W, self.__w_c
        yield self.SOFTMAX_W, self.__w_l
        yield self.ASPECT_B, self.__b_a
        yield self.CONTEXT_B, self.__b_c
        yield self.SOFTMAX_B, self.__b_l

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(IANFrames, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u'aspect_att', self.__aspect_att
        yield u'context_att', self.__context_att
        yield u'aspects', self.AspectInput

    # endregion

    # region public 'create' methods

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IANFrames, self).create_feed_dict(input=input, data_type=data_type)
        feed_dict[self.__dropout_rnn_keep_prob] = self.Config.DropoutRNNKeepProb if data_type == DataType.Train else 1.0
        return feed_dict

    # endregion

    # region private methods

    @staticmethod
    def __aggreagate(config, outputs, length):
        assert(isinstance(config, IANFramesConfig))
        if config.StatesAggregationMode == StatesAggregationModes.AVERAGE:
            return tf.reduce_mean(outputs, 1)
        if config.StatesAggregationMode == StatesAggregationModes.LAST_IN_SEQUENCE:
            return core.networks.tf_helpers.sequence.select_last_relevant_in_sequence(outputs, length)
        else:
            raise Exception('"{}" type does not supported'.format(config.StatesAggregationMode))

    def __compose_all_parameters(self):
        e_term_type_indices = tf.constant(value=1.0,
                                          shape=[self.Config.BatchSize,
                                                 self.Config.MaxAspectLength,
                                                 self.TermTypeEmbeddingSize])

        aspect_input = tf.concat(values=self.__get_embedded_parameters() + [e_term_type_indices],
                                 axis=-1)

        aspect_embedded = self.process_embedded_data(
            embedded=aspect_input,
            dropout_keep_prob=self.EmbeddingDropoutKeepProb)

        aspect_embedded = tf.reshape(aspect_embedded, [self.Config.BagsPerMinibatch,
                                                       self.Config.MaxAspectLength,
                                                       self.TermEmbeddingSize])

        return aspect_embedded

    def __get_embedded_parameters(self):
        embedded_params = []
        for e, v in embedding.get_ev(self):
            param_inds = filter_batch_elements(elements=v,
                                               inds=self.AspectInput,
                                               handler=select_entity_related_elements)
            emb_param = tf.nn.embedding_lookup(e, param_inds)
            embedded_params.append(emb_param)

        return embedded_params

    # endregion
