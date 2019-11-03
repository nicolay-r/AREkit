from core.networks.context.configurations.att_hidden_bilstm import AttentionHiddenBiLSTMConfig


class AttentionHiddenZYangBiLSTMConfig(AttentionHiddenBiLSTMConfig):

    __attention_size = 100

    def __init__(self):
        super(AttentionHiddenZYangBiLSTMConfig, self).__init__()

    @property
    def AttentionSize(self):
        return self.__attention_size
