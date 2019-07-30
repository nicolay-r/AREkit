from core.networks.context.configurations.bi_lstm import BiLSTMConfig


class AttBiLSTMConfig(BiLSTMConfig):
    """
    Authors: SeoSangwoo
    Paper: https://www.aclweb.org/anthology/P16-2034
    Repository: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    """

    def __init__(self):
        super(AttBiLSTMConfig, self).__init__()
