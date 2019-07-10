from core.networks.attention.architectures.yatian import AttentionYatianColing2016
from core.networks.attention.configurations.yatian import AttentionYatianColing2016Config
from core.networks.context.configurations.cnn import CNNConfig


class AttentionCNNConfig(CNNConfig):

    def __init__(self, attention_config):
        assert(isinstance(attention_config, AttentionYatianColing2016Config))
        super(AttentionCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = attention_config

    def AttentionModel(self):
        return self.__attention

    def notify_initialization_completed(self):
        assert(self.__attention_model is None)

        self.__attention_model = AttentionYatianColing2016(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            term_embedding_size=self.TermEmbeddingShape[1])

    def _internal_get_parameters(self):
        parameters = super(AttentionCNNConfig, self)._internal_get_parameters()
        parameters += self.__attention_config.get_parameters()
        return parameters
