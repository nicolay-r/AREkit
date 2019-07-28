from core.networks.context.configurations.bi_lstm import BiLSTMConfig


class SelfAttentionBiLSTMConfig(BiLSTMConfig):
    """
    Paper: https://arxiv.org/abs/1703.03130
    Code Author: roomylee, https://github.com/roomylee (C)
    Code: https://github.com/roomylee/self-attentive-emb-tf
    """

    def __init__(self):
        super(SelfAttentionBiLSTMConfig, self).__init__()
        self.__fc_size = 300
        self.__r_size = 3
        self.__d_a_size = 0
        self.__p_coef = 0

    @property
    def FullyConnectionSize(self):
        return self.__fc_size

    @property
    def RSize(self):
        return self.__r_size

    @property
    def PCoef(self):
        return self.__p_coef

    @property
    def DASize(self):
        return self.__d_a_size
