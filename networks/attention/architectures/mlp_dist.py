import tensorflow as tf
from core.networks.attention.architectures.mlp import MultiLayerPerceptronAttention
from core.networks.attention.configurations.mlp_dist import MLPWithDistancesAttentionConfig


class MLPWithDistances(MultiLayerPerceptronAttention):
    """
    Authors: Yatian Shen, Xuanjing Huang
    Implement modified MLP attention with distance vectors in input.
    Check that originally in MLP distances has not been supported.
    However this is not significant breakthrough, but still we should try to do so.
    """

    def __init__(self, cfg, batch_size, terms_per_context, term_embedding_size):
        assert(isinstance(cfg, MLPWithDistancesAttentionConfig))
        # TODO. Implement modified MLP attention with distance vectors in input.
        super(MLPWithDistances, self).__init__(
            cfg=cfg,
            batch_size=batch_size,
            terms_per_context=terms_per_context,
            term_embedding_size=term_embedding_size + cfg.DistanceEmbeddingSize)

    def create_embedding(self, embeddings):
        assert(isinstance(embeddings, dict))
        term_embedding = super(MLPWithDistances, self).create_embedding(embeddings)
        # TODO. Add distance embedding.
        distance_embedding = []
        return

    def set_input(self, dict_params):
        # TODO. Provide dist parameter and use the dict.
        pass

    def init_input(self):
        super(MLPWithDistances, self).init_input()
        # TODO. Add Dist.
