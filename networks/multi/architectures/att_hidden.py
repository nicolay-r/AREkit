from collections import OrderedDict

import tensorflow as tf
from core.networks.attention.architectures.rnn_attention_p_zhou import attention_by_peng_zhou
from core.networks.multi.architectures.base import BaseMultiInstanceNeuralNetwork


# TODO. Complete.
class AttHidden(BaseMultiInstanceNeuralNetwork):
    """
    Utilize attention in rnn_attention_based architectures.
    """

    def __init__(self, context_network):
        super(AttHidden, self).__init__(context_network)
        self.__hidden = OrderedDict()

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: Tensor
            [batches, sentences, embedding]
        """
        with tf.variable_scope('mi_attention'):
            attn, self.__att_alphas = attention_by_peng_zhou(context_outputs)

    def iter_hidden_parameters(self):
        for name, value in self.__hidden.iteritems():
            yield name, value

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttHidden, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_alphas

