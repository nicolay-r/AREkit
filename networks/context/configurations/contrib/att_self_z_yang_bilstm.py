from arekit.networks.context.configurations.contrib.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig


class AttentionSelfZYangBiLSTMConfig(AttentionSelfPZhouBiLSTMConfig):

    __attention_size = 100

    def __init__(self):
        super(AttentionSelfZYangBiLSTMConfig, self).__init__()

    @property
    def AttentionSize(self):
        return self.__attention_size
