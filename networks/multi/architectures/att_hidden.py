import tensorflow as tf
from core.networks.attention.architectures.rnn_attention_p_zhou import attention_by_peng_zhou
from core.networks.multi.architectures.base_single_mlp import BaseMultiInstanceSingleMLP


class AttHiddenOverSentences(BaseMultiInstanceSingleMLP):
    """
    Utilize sequence-based attention architectures.
    """

    def __init__(self, context_network):
        super(AttHiddenOverSentences, self).__init__(context_network)
        self.__att_alphas = None

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: Tensor
            [batches, sentences, embedding]
        """
        context_outputs = tf.reshape(context_outputs, [self.Config.BatchSize,
                                                       self.Config.BagSize,
                                                       self.ContextNetwork.ContextEmbeddingSize])

        with tf.variable_scope('mi_attention'):
            att_output, self.__att_alphas = attention_by_peng_zhou(context_outputs)

        return att_output

    # region 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttHiddenOverSentences, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_alphas

    # endregion
