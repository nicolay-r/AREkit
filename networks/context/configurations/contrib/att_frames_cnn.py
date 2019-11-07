from core.networks.attention.architectures.cnn_attention_mlp_interactive import InteractiveMLPAttention
from core.networks.attention.configurations.cnn_attention_mlp_dynamic import MultiLayerPerceptronAttentionDynamicConfig
from core.networks.context.configurations.cnn import CNNConfig


class AttentionFramesCNNConfig(CNNConfig):

    def __init__(self):
        super(AttentionFramesCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = MultiLayerPerceptronAttentionDynamicConfig()

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def notify_initialization_completed(self):
        assert(self.__attention is None)

        self.__attention = InteractiveMLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            term_embedding_size=self.TermEmbeddingShape[1],
            pos_embedding_size=self.PosEmbeddingSize,
            dist_embedding_size=self.DistanceEmbeddingSize)

    def _internal_get_parameters(self):
        parameters = super(AttentionFramesCNNConfig, self)._internal_get_parameters()
        parameters += self.__attention_config.get_parameters()
        return parameters

    # endregion