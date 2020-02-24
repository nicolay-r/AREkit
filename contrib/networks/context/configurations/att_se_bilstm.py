from arekit.contrib.networks.context.configurations.base.att_bilstm_base import AttentionBiLSTMBaseConfig


class AttentionSynonymEndsBiLSTMConfig(AttentionBiLSTMBaseConfig):

    def __init__(self):
        super(AttentionSynonymEndsBiLSTMConfig, self).__init__(
            keys_count=self.SynonymsPerContext * 2,
            att_support_zero_length=False)
