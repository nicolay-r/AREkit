from core.networks.attention.architectures.mlp import MultiLayerPerceptronAttention
from core.networks.attention.configurations.mlp import MultiLayerPerceptronAttentionConfig
from core.networks.context.configurations.cnn import CNNConfig


class AttentionCNNConfig(CNNConfig):

    def __init__(self):
        super(AttentionCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = MultiLayerPerceptronAttentionConfig()

    @property
    def AttentionModel(self):
        return self.__attention

    def notify_initialization_completed(self):
        assert(self.__attention is None)

        self.__attention = MultiLayerPerceptronAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            term_embedding_size=self.TermEmbeddingShape[1],
            pos_embedding_size=self.PosEmbeddingSize,
            dist_embedding_size=self.DistanceEmbeddingSize)

    def _internal_get_parameters(self):
        parameters = super(AttentionCNNConfig, self)._internal_get_parameters()
        parameters += self.__attention_config.get_parameters()
        return parameters
