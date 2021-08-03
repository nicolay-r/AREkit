from arekit.contrib.networks.context.configurations.base.att_bilstm_base import AttentionBiLSTMBaseConfig


class AttentionEndsAndFramesBiLSTMConfig(AttentionBiLSTMBaseConfig):

    def __init__(self):
        super(AttentionEndsAndFramesBiLSTMConfig, self).__init__(
            keys_count=self.FramesPerContext + 2,
            att_support_zero_length=False)
