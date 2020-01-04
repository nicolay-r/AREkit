from arekit.networks.attention.architectures.mlp_interactive import InteractiveMLPAttention
from arekit.networks.attention.configurations.mlp_interactive import InteractiveMLPAttentionConfig
from arekit.networks.context.configurations.att_cnn_base import AttentionCNNBaseConfig


class AttentionSynonymEndsCNNConfig(AttentionCNNBaseConfig):

    def __init__(self):
        super(AttentionSynonymEndsCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = InteractiveMLPAttentionConfig(self.SynonymsPerContext * 2)

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def get_attention_parameters(self):
        return self.__attention_config

    def notify_initialization_completed(self):
        assert(self.__attention is None)

        self.__attention = InteractiveMLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            support_zero_length=False)

    # endregion
