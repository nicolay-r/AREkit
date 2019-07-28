from core.networks.context.configurations.bi_lstm import BiLSTMConfig


class SelfAttentionBiLSTMConfig(BiLSTMConfig):
    """
    Paper: https://arxiv.org/abs/1703.03130
    Code Author: roomylee, https://github.com/roomylee (C)
    Code: https://github.com/roomylee/self-attentive-emb-tf
    """

    def __init__(self):
        super(SelfAttentionBiLSTMConfig, self).__init__()
