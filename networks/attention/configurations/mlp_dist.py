from core.networks.attention.configurations.mlp import MultiLayerPerceptronAttentionConfig


class MLPWithDistancesAttentionConfig(MultiLayerPerceptronAttentionConfig):

    __distance_embedding_size = 5

    @property
    def DistanceEmbeddingSize(self):
        return self.__distance_embedding_size

    def get_parameters(self):
        parameters = super(MLPWithDistancesAttentionConfig, self).get_parameters()
        parameters.append(
            ("attention-yatian-coling-2016:distance-embedding-size", self.DistanceEmbeddingSize)
        )

        return parameters
