import tensorflow as tf
from core.networks.attention.architectures.cnn_attention_mlp import MultiLayerPerceptronAttention
from core.networks.context.sample import InputSample
from core.networks.tf_helpers import filtering
from core.networks.tf_helpers import sequence


class MultiLayerPerceptronAttentionDynamic(MultiLayerPerceptronAttention):
    """
    Averaging result attention of MultiLayerPerceptronAttention
    """

    def __init__(self, cfg,
                 batch_size,
                 terms_per_context,
                 term_embedding_size,
                 pos_embedding_size,
                 dist_embedding_size):

        super(MultiLayerPerceptronAttentionDynamic, self).__init__(
            cfg,
            batch_size,
            terms_per_context,
            term_embedding_size,
            pos_embedding_size,
            dist_embedding_size)

        self.__dynamic_lens = None

    @property
    def AttentionEmbeddingSize(self):
        """
        Considered single entity as an average of the result
        """
        return 1 * self.TermEmbeddingSize

    def calculate_entities_length_func(self, entities):
        scalar_length = super(MultiLayerPerceptronAttentionDynamic, self).calculate_entities_length_func(entities)

        self.__dynamic_lens = sequence.calculate_sequence_length(
            sequence=entities,
            is_neg_placeholder=InputSample.FRAMES_PAD_VALUE < 0)

        return scalar_length

    def reshape_att_sum(self, att_sum):
        """
        att_sum: [batch_size, entity_per_context, term_embedding_size]
        """
        _att_sum = tf.reshape(att_sum, shape=[self.BatchSize,
                                              self.Config.EntitiesPerContext,
                                              self.TermEmbeddingSize])

        mean_sum = filtering.filter_batch_elements(
            elements_type=tf.float32,
            elements=_att_sum,
            inds=self.__dynamic_lens,
            handler=self.crop_elements_by_lengths_and_reduce_mean)

        return super(MultiLayerPerceptronAttentionDynamic, self).reshape_att_sum(mean_sum)

    def reshape_att_weights(self, att_weights):
        """
        att_sum: [batch_size, entity_per_context, terms_per_context]
        """
        _att_weights = tf.reshape(att_weights, shape=[self.BatchSize,
                                                      self.Config.EntitiesPerContext,
                                                      self.TermEmbeddingSize])

        mean_sum = filtering.filter_batch_elements(
            elements_type=tf.float32,
            elements=_att_weights,
            inds=self.__dynamic_lens,
            handler=self.crop_elements_by_lengths_and_reduce_mean)

        return tf.reshape(mean_sum, shape=[self.BatchSize, self.TermEmbeddingSize])

    @staticmethod
    def crop_elements_by_lengths_and_reduce_mean(i, elements, lens, filtered):
        """
        Crop all elements by calculated length and perform reduce_mean operation.

        elements: [batch, entity_per_context, term_embedding_size]
        lens: [batch, 1]
        """
        row_elements = tf.squeeze(tf.gather(elements, [i], axis=0))  # [entity_per_context, term_embedding_size]
        term_embedding_size = row_elements.shape[-1]
        row_len_original = tf.reshape(tf.gather(lens, [i], axis=0), [])  # scalar
        row_len_non_zero = tf.maximum(row_len_original, 1)
        row_frames = tf.slice(row_elements, begin=[0, 0], size=[row_len_non_zero, term_embedding_size])
        is_not_empty = tf.sign(row_len_original)

        mean = tf.reduce_mean(row_frames, axis=0)   # result: [terms_embedding_size]
        result = tf.multiply(x=tf.cast(is_not_empty, dtype=tf.float32),
                             y=mean)   # 0 -- in case of empty sequence original, otherwize the same

        result = tf.reshape(result, shape=[term_embedding_size])

        return (i + 1,
                elements,
                lens,
                filtered.write(i, tf.squeeze(result)))
