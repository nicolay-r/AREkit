from arekit.networks.context.configurations.att_bilstm_base import AttentionBiLSTMBaseConfig


class AttentionSynonymEndsAndFramesBiLSTMConfig(AttentionBiLSTMBaseConfig):

    def __init__(self):
        super(AttentionSynonymEndsAndFramesBiLSTMConfig, self).__init__(
            keys_count=self.FramesPerContext + 2 * self.SynonymsPerContext,
            att_support_zero_length=False)
