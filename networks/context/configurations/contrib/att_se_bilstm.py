from arekit.networks.context.configurations.att_bilstm_base import AttentionBiLSTMBaseConfig


class AttentionSynonymEndsBiLSTMConfig(AttentionBiLSTMBaseConfig):

    def __init__(self):
        super(AttentionSynonymEndsBiLSTMConfig, self).__init__(keys_count=self.SynonymsPerContext * 2)
