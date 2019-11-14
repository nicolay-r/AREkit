import tensorflow as tf
from core.networks.attention.architectures.rnn_attention_p_zhou import attention_by_peng_zhou
from core.networks.context.architectures.bilstm import BiLSTM


class AttentionHiddenBiLSTM(BiLSTM):
    """
    Authors: Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
    Paper: https://www.aclweb.org/anthology/P16-2034
    Code author: SeoSangwoo (c), https://github.com/SeoSangwoo
    Code: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

    NOTE: We consider 'hidden' since attention utilize a hidden states as keys towards
    embedded context.
    """

    def __init__(self):
        super(AttentionHiddenBiLSTM, self).__init__()
        self.__att_alphas = None

    def get_attention_output_with_alphas(self, rnn_outputs):
        return attention_by_peng_zhou(rnn_outputs)

    def customize_rnn_output(self, rnn_outputs, s_length):
        with tf.variable_scope("attention"):
            att_output, self.__att_alphas = self.get_attention_output_with_alphas(rnn_outputs)

        return att_output

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionHiddenBiLSTM, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_alphas
