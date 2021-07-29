from arekit.contrib.networks.context.configurations.base.att_bilstm_base import AttentionBiLSTMBaseConfig


class AttentionFramesBiLSTMConfig(AttentionBiLSTMBaseConfig):

    def __init__(self):
        super(AttentionFramesBiLSTMConfig, self).__init__(
            keys_count=self.FramesPerContext,
            att_support_zero_length=True)
