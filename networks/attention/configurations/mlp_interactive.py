from arekit.networks.attention.configurations.mlp import MLPAttentionConfig


class InteractiveMLPAttentionConfig(MLPAttentionConfig):

    def __init__(self, keys_count):
        assert(isinstance(keys_count, int))
        assert(keys_count >= 0)
        self.__keys_count = keys_count

    @property
    def EntitiesPerContext(self):
        return self.__keys_count
