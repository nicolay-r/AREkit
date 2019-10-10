import tensorflow as tf
from tensorflow.python.ops import math_ops

from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.sequence import get_cell
from core.networks.context.configurations.ian import IANConfig
from core.networks.context.configurations.rnn import CellTypes
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
        self.__E_aspects = tf.get_variable(name="E_aspects",
                                           dtype=tf.float32,
                                           initializer=tf.random_normal_initializer,
                                           shape=self.Config.AspectsEmbeddingShape,
                                           trainable=True)

    def init_hidden_states(self):
        assert(isinstance(self.Config, IANConfig))

        with tf.name_scope('weights'):
            self.__weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.Config.HiddenSize, self.Config.HiddenSize],
                    initializer=self.Config.WeightInitializer,
                    regularizer=self.Config.LayerRegularizer
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.Config.HiddenSize, self.Config.HiddenSize],
                    initializer=self.Config.WeightInitializer,
                    regularizer=self.Config.LayerRegularizer
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.ContextEmbeddingSize, self.Config.ClassesCount],
                    initializer=self.Config.WeightInitializer,
                    regularizer=self.Config.LayerRegularizer
                ),
            }

        with tf.name_scope('biases'):
            self.__biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.Config.MaxAspectLength, 1],
                    initializer=self.Config.BiasInitializer,
                    regularizer=self.Config.LayerRegularizer
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.Config.MaxContextLength, 1],
                    initializer=self.Config.BiasInitializer,
                    regularizer=self.Config.LayerRegularizer
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.Config.ClassesCount],
                    initializer=self.Config.BiasInitializer,
                    regularizer=self.Config.LayerRegularizer
                ),
            }

    def init_embedded_input(self):
        aspect_inputs = tf.cast(tf.nn.embedding_lookup(self.__E_aspects, self.__aspects), tf.float32)
        self.__aspect_inputs = self.optional_process_embedded_data(
            self.Config,
            aspect_inputs,
            self.EmbeddingDropoutKeepProb)

        return super(IAN, self).init_embedded_input()

    def init_context_embedding(self, embedded_terms):
        with tf.name_scope('inputs'):
            context_inputs = embedded_terms
            aspect_inputs = self.__aspect_inputs

        with tf.name_scope('dynamic_rnn'):
            aspect_lens = utils.calculate_sequence_length(self.__aspects)

            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                cell=get_cell(hidden_size=self.Config.HiddenSize, cell_type=self.Config.CellType),
                inputs=aspect_inputs,
                sequence_length=aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )

            aspect_avg = tf.reduce_mean(aspect_outputs, 1)

            context_lens = utils.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))

            context_outputs, context_state = tf.nn.dynamic_rnn(
                cell=get_cell(hidden_size=self.Config.HiddenSize, cell_type=self.Config.CellType),
                inputs=context_inputs,
                sequence_length=context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            context_avg = tf.reduce_mean(context_outputs, 1)

            aspect_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter = aspect_outputs_iter.unstack(aspect_outputs)
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(context_avg)
            aspect_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            aspect_lens_iter = aspect_lens_iter.unstack(aspect_lens)
            aspect_rep = tf.TensorArray(size=self.Config.BatchSize, dtype=tf.float32)
            aspect_att = tf.TensorArray(size=self.Config.BatchSize, dtype=tf.float32)

            def aspect_body(i, aspect_rep, aspect_att, weights, biases):
                a = aspect_outputs_iter.read(i)
                b = context_avg_iter.read(i)
                l = math_ops.to_int32(aspect_lens_iter.read(i))
                aspect_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a,
                                        weights['aspect_score']),
                                        tf.reshape(b, [-1, 1])) + biases['aspect_score']),
                                        [1, -1])
                aspect_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(aspect_score, [0, 0], [1, l])),
                     tf.zeros([1, self.Config.MaxAspectLength - l])],
                     1)
                aspect_att = aspect_att.write(i, aspect_att_temp)
                aspect_rep = aspect_rep.write(i, tf.matmul(aspect_att_temp, a))
                return (i + 1, aspect_rep, aspect_att, weights, biases)

            def condition(i, aspect_rep, aspect_att, weights, biases):
                return i < self.Config.BatchSize

            _, aspect_rep_final, aspect_att_final, _, _ = tf.while_loop(
                cond=condition,
                body=aspect_body,
                loop_vars=(0, aspect_rep, aspect_att, self.__weights, self.__biases))

            self.__aspect_atts = tf.reshape(aspect_att_final.stack(), [-1, self.Config.MaxAspectLength])
            self.__aspect_reps = tf.reshape(aspect_rep_final.stack(), [-1, self.Config.HiddenSize])

            context_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter = context_outputs_iter.unstack(context_outputs)
            aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            context_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            context_lens_iter = context_lens_iter.unstack(context_lens)
            context_rep = tf.TensorArray(size=self.Config.BatchSize, dtype=tf.float32)
            context_att = tf.TensorArray(size=self.Config.BatchSize, dtype=tf.float32)

            def context_body(i, context_rep, context_att, weights, biases):
                a = context_outputs_iter.read(i)
                b = aspect_avg_iter.read(i)
                l = math_ops.to_int32(context_lens_iter.read(i))
                context_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, self.__weights['context_score']), tf.reshape(b, [-1, 1])) + self.__biases['context_score']), [1, -1])
                context_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(context_score, [0, 0], [1, l])), tf.zeros([1, self.Config.MaxContextLength - l])],
                    1)
                context_att = context_att.write(i, context_att_temp)
                context_rep = context_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_rep, context_att, weights, biases)

            def condition(i, context_rep, context_att, weights, biases):
                return i < self.Config.BatchSize

            _, context_rep_final, context_att_final, _, _ = tf.while_loop(
                cond=condition,
                body=context_body,
                loop_vars=(0, context_rep, context_att, self.__weights, self.__biases))

            self.__context_atts = tf.reshape(context_att_final.stack(), [-1, self.Config.MaxContextLength])
            self.__context_reps = tf.reshape(context_rep_final.stack(), [-1, self.Config.HiddenSize])

            return tf.concat([self.__aspect_reps, self.__context_reps], 1)

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

        feed_dict[self.__aspects] = input[InputSample.I_FRAME_INDS]

        return feed_dict
