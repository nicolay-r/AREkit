import logging
import tensorflow as tf
from arekit.networks.attention.architectures.mlp import MLPAttention
from arekit.networks.context.sample import InputSample
from arekit.networks.tf_helpers import filtering
from arekit.networks.tf_helpers import sequence


logger = logging.getLogger(__name__)


class InteractiveMLPAttention(MLPAttention):
    """
    Averaging result attention of MultiLayerPerceptronAttention
    """

    def __init__(self, cfg,
                 batch_size,
                 terms_per_context,
                 support_zero_length):
        super(InteractiveMLPAttention, self).__init__(cfg, batch_size, terms_per_context)
        assert(isinstance(support_zero_length, bool))

        self.__dynamic_lens = None
        self.__support_zero_length = support_zero_length

    @property
    def AttentionEmbeddingSize(self):
        """
        Considered single entity as an average of the result
        """
        return self.TermEmbeddingSize

    @property
    def Handler(self):
        if self.__support_zero_length:
            return InteractiveMLPAttention.crop_elements_by_lengths_and_reduce_mean_with_zero_level_support
        else:
            return InteractiveMLPAttention.crop_elements_by_lengths_and_reduce_mean

    def calculate_keys_length_func(self, keys):
        scalar_length = super(InteractiveMLPAttention, self).calculate_keys_length_func(keys)

        self.__dynamic_lens = sequence.calculate_sequence_length(
            sequence=keys,
            is_neg_placeholder=InputSample.FRAMES_PAD_VALUE < 0)

        return scalar_length

    def reshape_att_sum(self, att_sum):
        """
        att_sum: [batch_size, entity_per_context, term_embedding_size]
        """
        _att_sum = tf.reshape(att_sum, shape=[self.BatchSize,
                                              self.Config.KeysPerContext,
                                              self.TermEmbeddingSize])

        mean_sum = filtering.filter_batch_elements(
            elements_type=tf.float32,
            elements=_att_sum,
            inds=self.__dynamic_lens,
            handler=self.Handler)

        return super(InteractiveMLPAttention, self).reshape_att_sum(mean_sum)

    def reshape_att_weights(self, att_weights):
        """
        att_sum: [batch_size, entity_per_context, terms_per_context]
        """
        _att_weights = tf.reshape(att_weights, shape=[self.BatchSize,
                                                      self.Config.KeysPerContext,
                                                      self.TermsPerContext])

        mean_sum = filtering.filter_batch_elements(
            elements_type=tf.float32,
            elements=_att_weights,
            inds=self.__dynamic_lens,
            handler=self.Handler)

        return tf.reshape(mean_sum, shape=[self.BatchSize, self.TermsPerContext])

    @staticmethod
    def __core_process_batch_element(i, elements, lens):
        """
        Crop all elements by calculated length and perform reduce_mean operation.

        elements: [batch, entity_per_context, term_embedding_size]
        lens: [batch, 1]
        extra_parameter: bool
            used for flag support_zero_length
        """
        row_elements = tf.squeeze(tf.gather(elements, [i], axis=0))  # [entity_per_context, term_embedding_size]
        term_embedding_size = row_elements.shape[-1]
        row_len_original = tf.reshape(tf.gather(lens, [i], axis=0), [])  # scalar
        row_frames = tf.slice(row_elements, begin=[0, 0], size=[row_len_original, term_embedding_size])

        return row_frames, term_embedding_size, row_len_original

    @staticmethod
    def crop_elements_by_lengths_and_reduce_mean_with_zero_level_support(i, elements, lens, filtered):
        """
        This handler supports the case of empty list of elements by providing a normal_dist vector in case of
        an empty amount of elements.

        It could be utilized for frames, as amount of frames per context may vary.
        The latter might not be presented in context.
        """
        logger.info("Compile model with logger which supports lens == 0")

        row_frames, term_embedding_size, row_len_original = InteractiveMLPAttention.__core_process_batch_element(
            i=i,
            elements=elements,
            lens=lens)

        zeros = tf.zeros(shape=[1, term_embedding_size])
        non_empty_frames = tf.concat(values=[row_frames, zeros], axis=0)

        # This is a correction as we have a following equation:
        # l -- denotes a row_length_original
        #    x      y         l+1
        #    -  == --- => y = --- x
        #    l     l+1         l

        scalar = tf.divide(tf.add(row_len_original, 1), tf.maximum(row_len_original, 1))
        scalar = tf.cast(scalar, dtype=tf.float32)
        row_frames = tf.scalar_mul(scalar=scalar, x=non_empty_frames)

        mean = tf.reduce_mean(row_frames, axis=0)  # result: [terms_embedding_size]
        result = tf.reshape(mean, shape=[term_embedding_size])

        return (i + 1,
                elements,
                lens,
                filtered.write(i, tf.squeeze(result)))

    @staticmethod
    def crop_elements_by_lengths_and_reduce_mean(i, elements, lens, filtered):
        """
        This is a general handler.

        Handler with the following limitation:
        - lens values > 0
        """
        logger.info("Compile model with logger which has a limitation (lens > 0)")

        row_frames, term_embedding_size, _ = InteractiveMLPAttention.__core_process_batch_element(
            i=i, elements=elements, lens=lens)

        mean = tf.reduce_mean(row_frames, axis=0)  # result: [terms_embedding_size]
        result = tf.reshape(mean, shape=[term_embedding_size])

        return (i + 1,
                elements,
                lens,
                filtered.write(i, tf.squeeze(result)))
