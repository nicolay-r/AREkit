#!/usr/bin/python
import tensorflow as tf
from core.networks.context.architectures.cnn import VanillaCNN
from core.networks.context.configurations.cnn import CNNConfig
import utils


class PiecewiseCNN(VanillaCNN):

    @property
    def ContextEmbeddingSize(self):
        return 3 * self.Config.FiltersCount + \
               self._get_attention_vector_size(self.Config)

    def init_context_embedding(self, embedded_terms):
        embedded_terms = self.padding(embedded_terms, self.Config.WindowSize)

        bwc_line = tf.reshape(embedded_terms,
                              [self.Config.BatchSize,
                               (self.Config.TermsPerContext + (self.Config.WindowSize - 1)) * self.TermEmbeddingSize,
                               1])

        bwc_conv = tf.nn.conv1d(bwc_line, self.conv_filter, self.TermEmbeddingSize,
                                "VALID",
                                data_format="NHWC",
                                name="conv")

        # slice all data into 3 parts -- before, inner, and after according to relation
        sliced = tf.TensorArray(dtype=tf.float32, size=self.Config.BatchSize, infer_shape=False, dynamic_size=True)
        _, _, _, _, _, sliced = tf.while_loop(
                lambda i, *_: tf.less(i, self.Config.BatchSize),
                self.splitting,
                [0, self.InputPSubjInd, self.InputPObjInd, bwc_conv, self.Config.FiltersCount, sliced])
        sliced = tf.squeeze(sliced.concat())

        # Max Pooling
        bwgc_mpool = tf.nn.max_pool(
                sliced,
                [1, 1, self.Config.TermsPerContext, 1],
                [1, 1, self.Config.TermsPerContext, 1],
                padding='VALID',
                data_format="NHWC")

        bwc_mpool = tf.squeeze(bwgc_mpool, [2])
        bcw_mpool = tf.transpose(bwc_mpool, perm=[0, 2, 1])
        g = tf.reshape(bcw_mpool, [self.Config.BatchSize, 3 * self.Config.FiltersCount])

        if self.Config.UseAttention:
            # TODO. in Nested class, as it is specific att application.
            g = tf.concat([g, self.init_attention_embedding()], axis=1)

        return tf.concat(g, axis=-1)

    def init_logits_unscaled(self, context_embedding):
        return utils.get_two_layer_logits(
            g=context_embedding,
            W1=self.W,
            b1=self.b,
            W2=self.W2,
            b2=self.b2,
            dropout_keep_prob=self.dropout_keep_prob,
            activations=[tf.tanh, tf.tanh, None])

    def init_hidden_states(self):
        assert(isinstance(self.Config, CNNConfig))
        super(PiecewiseCNN, self).init_hidden_states()

        self.W = tf.Variable(initial_value=tf.random_normal([self.ContextEmbeddingSize, self.Config.HiddenSize]),
                             dtype=tf.float32,
                             name="W")

    @staticmethod
    def splitting(i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs):
        l_ind = tf.minimum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # left
        r_ind = tf.maximum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # right

        w = tf.Variable(bwc_conv.shape[1], dtype=tf.int32) # total width (words count)

        b_slice_from = [i, 0, 0]
        b_slice_size = tf.concat([[1], l_ind, [channels_count]], 0)
        m_slice_from = tf.concat([[i], l_ind, [0]], 0)
        m_slice_size = tf.concat([[1], r_ind - l_ind, [channels_count]], 0)
        a_slice_from = tf.concat([[i], r_ind, [0]], 0)
        a_slice_size = tf.concat([[1], w-r_ind, [channels_count]], 0)

        bwc_split_b = tf.slice(bwc_conv, b_slice_from, b_slice_size)
        bwc_split_m = tf.slice(bwc_conv, m_slice_from, m_slice_size)
        bwc_split_a = tf.slice(bwc_conv, a_slice_from, a_slice_size)

        pad_b = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w-l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_m = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w-r_ind+l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_a = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([r_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        bwc_split_b = tf.pad(bwc_split_b, pad_b, constant_values=tf.float32.min)
        bwc_split_m = tf.pad(bwc_split_m, pad_m, constant_values=tf.float32.min)
        bwc_split_a = tf.pad(bwc_split_a, pad_a, constant_values=tf.float32.min)

        outputs = outputs.write(i, [[bwc_split_b, bwc_split_m, bwc_split_a]])

        i += 1
        return i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs
