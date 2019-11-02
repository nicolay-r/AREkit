from core.networks.attention.configurations.cnn_attention_mlp import MultiLayerPerceptronAttentionConfig


class MultiLayerPerceptronAttentionDynamicConfig(MultiLayerPerceptronAttentionConfig):

    __frames_per_context = 3

    @property
    def EntitiesPerContext(self):
        return self.__frames_per_context
