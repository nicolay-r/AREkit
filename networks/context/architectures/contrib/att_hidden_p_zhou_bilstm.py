from arekit.networks.attention.architectures.rnn_attention_p_zhou import attention_by_peng_zhou
from arekit.networks.context.architectures.att_hidden_bilstm_base import AttentionHiddenBiLSTMBase


class AttentionHiddenPZhouBiLSTM(AttentionHiddenBiLSTMBase):
    """
    Authors: Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
    Paper: https://www.aclweb.org/anthology/P16-2034
    Code author: SeoSangwoo (c), https://github.com/SeoSangwoo
    Code: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    """

    def get_attention_output_with_alphas(self, rnn_outputs):
        return attention_by_peng_zhou(rnn_outputs)
