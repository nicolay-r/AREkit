from core.networks.attention.architectures.cnn_attention_mlp import MLPAttention
from core.networks.attention.configurations.cnn_attention_mlp import MultiLayerPerceptronAttentionConfig
from core.networks.context.configurations.cnn import CNNConfig


class AttentionAttitudeEndsCNNConfig(CNNConfig):

    def __init__(self):
        super(AttentionAttitudeEndsCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = MultiLayerPerceptronAttentionConfig()

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def notify_initialization_completed(self):
        assert(self.__attention is None)

        self.__attention = MLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext)

    def _internal_get_parameters(self):
        parameters = super(AttentionAttitudeEndsCNNConfig, self)._internal_get_parameters()
        parameters += self.__attention_config.get_parameters()
        return parameters

    # endregion
