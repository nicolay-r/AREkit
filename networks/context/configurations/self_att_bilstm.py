import tensorflow as tf
from core.networks.context.configurations.bi_lstm import BiLSTMConfig


class SelfAttentionBiLSTMConfig(BiLSTMConfig):
    """
    Paper: https://arxiv.org/abs/1703.03130
    Code Author: roomylee, https://github.com/roomylee (C)
    Code: https://github.com/roomylee/self-attentive-emb-tf
    """

    def __init__(self):
        """
        d_a_size: int
            Size of W_s1 embedding
        r_size: int
            Size of W_s2 embedding
        fc_size: int
            Size of fully connected laye
        p_coef: int
            Coefficient for penalty
        """
        super(SelfAttentionBiLSTMConfig, self).__init__()
        super(SelfAttentionBiLSTMConfig, self).modify_weight_initializer(tf.contrib.layers.xavier_initializer())
        super(SelfAttentionBiLSTMConfig, self).modify_bias_initializer(tf.constant_initializer(0.1))
        super(SelfAttentionBiLSTMConfig, self).modify_learning_rate(0.1)

        self.__fc_size = 200
        self.__r_size = 30
        self.__d_a_size = 350
        self.__p_coef = 1.0

    # region public methods

    @property
    def FullyConnectionSize(self):
        return self.__fc_size

    @property
    def RSize(self):
        return self.__r_size

    @property
    def PenaltizationTermCoef(self):
        return self.__p_coef

    @property
    def DASize(self):
        return self.__d_a_size

    # endregion

    def modify_penaltization_term_coef(self, value):
        self.__p_coef = value

    def _internal_get_parameters(self):
        parameters = super(SelfAttentionBiLSTMConfig, self)._internal_get_parameters()

        parameters += [
            ("sa-bilstm:fully_connection_size", self.FullyConnectionSize),
            ("sa-bilstm:penalty_term_coef", self.PenaltizationTermCoef),
            ("sa-bilstm:da_size (size of ws_1)", self.DASize),
            ("sa-bilstm:r_size (size of ws_2)", self.RSize)
        ]

        return parameters
