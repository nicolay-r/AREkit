import tensorflow as tf

from arekit.networks.attention import common
from arekit.networks.attention.architectures.self_p_zhou import self_attention_by_peng_zhou
from arekit.networks.multi.architectures.base_single_mlp import BaseMultiInstanceSingleMLP


class AttSelfOverSentences(BaseMultiInstanceSingleMLP):
    """
    Utilize sequence-based attention architectures.
    """

    def __init__(self, context_network):
        super(AttSelfOverSentences, self).__init__(context_network)
        self.__att_alphas = None

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: Tensor
            [batches, sentences, embedding]
        """
        context_outputs = tf.reshape(context_outputs, [self.Config.BatchSize,
                                                       self.Config.BagSize,
                                                       self.ContextNetwork.ContextEmbeddingSize])

        with tf.variable_scope("mi_{}".format(common.ATTENTION_SCOPE_NAME)):
            att_output, self.__att_alphas = self_attention_by_peng_zhou(context_outputs)

        return att_output

    # region 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttSelfOverSentences, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion
