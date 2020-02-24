from arekit.contrib.networks.attention.architectures.self_p_zhou import self_attention_by_peng_zhou
from arekit.contrib.networks.context.architectures.base.att_self_bilstm_base import AttentionSelfBiLSTMBase


class AttentionSelfPZhouBiLSTM(AttentionSelfBiLSTMBase):
    """
    Authors: Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
    Paper: https://www.aclweb.org/anthology/P16-2034
    Code author: SeoSangwoo (c), https://github.com/SeoSangwoo
    Code: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    """

    def get_attention_output_with_alphas(self, rnn_outputs):
        return self_attention_by_peng_zhou(rnn_outputs)
