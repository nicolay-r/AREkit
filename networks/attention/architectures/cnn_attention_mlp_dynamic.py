import tensorflow as tf
from core.networks.attention.architectures.cnn_attention_mlp import MultiLayerPerceptronAttention
from core.networks.tf_helpers.sequence import calculate_sequence_length


class MultiLayerPerceptronAttentionDynamic(MultiLayerPerceptronAttention):
    """
    Averaging result attention of MultiLayerPerceptronAttention
    """

    @property
    def AttentionEmbeddingSize(self):
        """
        Considered single entity as an average of the result
        """
        return 1 * self.TermEmbeddingSize

    def calculate_entities_length_func(self, entities):
        # TODO. This should be scalar
        # TODO. Refactor and fix the bug.
        return calculate_sequence_length(
            sequence=entities,
            is_neg_placeholder=True)

    def reshape_att_sum(self, att_sum):
        """
        att_sum: [batch_size, entity_per_context, term_embedding_size]
        """
        _att_sum = tf.reduce_mean(att_sum, axis=1)
        return super(MultiLayerPerceptronAttentionDynamic, self).reshape_att_sum(_att_sum)

    def reshape_att_weights(self, att_weights):
        """
        att_sum: [batch_size, entity_per_context, terms_per_context]
        """
        _att_weights = tf.reduce_mean(att_weights, axis=1)
        _att_weights = tf.reshape(att_weights, shape=[self.BatchSize, self.TermsPerContext])
        return _att_weights
