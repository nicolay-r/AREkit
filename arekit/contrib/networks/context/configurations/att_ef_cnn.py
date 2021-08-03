from arekit.contrib.networks.attention.architectures.mlp_interactive import InteractiveMLPAttention
from arekit.contrib.networks.attention.configurations.mlp_interactive import InteractiveMLPAttentionConfig
from arekit.contrib.networks.context.configurations.base.att_cnn_base import AttentionCNNBaseConfig


class AttentionEndsAndFramesCNNConfig(AttentionCNNBaseConfig):

    def __init__(self):
        super(AttentionEndsAndFramesCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = InteractiveMLPAttentionConfig(self.FramesPerContext + 2)

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def get_attention_parameters(self):
        return self.__attention_config.get_parameters()

    def reinit_config_dependent_parameters(self):
        super(AttentionEndsAndFramesCNNConfig, self).reinit_config_dependent_parameters()

        self.__attention = InteractiveMLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            support_zero_length=False)

    # endregion
