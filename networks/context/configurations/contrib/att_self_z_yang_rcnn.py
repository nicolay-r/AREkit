from arekit.networks.context.configurations.rcnn import RCNNConfig


class AttentionSelfZYangRCNNConfig(RCNNConfig):

    __attention_size = 100

    def __init__(self):
        super(AttentionSelfZYangRCNNConfig, self).__init__()

    @property
    def AttentionSize(self):
        return self.__attention_size
