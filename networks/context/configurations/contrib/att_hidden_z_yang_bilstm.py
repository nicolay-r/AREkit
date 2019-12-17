from arekit.networks.context.configurations.contrib.att_hidden_p_zhou_bilstm import AttentionHiddenPZhouBiLSTMConfig


class AttentionHiddenZYangBiLSTMConfig(AttentionHiddenPZhouBiLSTMConfig):

    __attention_size = 100

    def __init__(self):
        super(AttentionHiddenZYangBiLSTMConfig, self).__init__()

    @property
    def AttentionSize(self):
        return self.__attention_size
