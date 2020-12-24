from arekit.contrib.networks.attention.architectures.mlp_interactive import InteractiveMLPAttention
from arekit.contrib.networks.attention.configurations.mlp_interactive import InteractiveMLPAttentionConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig


class AttentionBiLSTMBaseConfig(BiLSTMConfig):
    """
    Based on Interactive attention model
    """

    def __init__(self, keys_count, att_support_zero_length):
        super(AttentionBiLSTMBaseConfig, self).__init__()
        assert(isinstance(att_support_zero_length, bool))
        self.__attention = None
        self.__attention_config = InteractiveMLPAttentionConfig(keys_count=keys_count)
        self.__att_support_zero_length = att_support_zero_length

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def init_config_dependent_parameters(self):
        assert(self.__attention is None)
        super(AttentionBiLSTMBaseConfig, self).init_config_dependent_parameters()

        self.__attention = InteractiveMLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            support_zero_length=self.__att_support_zero_length)

    def _internal_get_parameters(self):
        parameters = super(AttentionBiLSTMBaseConfig, self)._internal_get_parameters()
        parameters += self.__attention_config.get_parameters()
        return parameters

    # endregion
