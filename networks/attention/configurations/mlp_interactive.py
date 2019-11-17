from core.networks.attention.configurations.mlp import MLPAttentionConfig


class InteractiveMLPAttentionConfig(MLPAttentionConfig):

    def __init__(self,  frames_count):
        assert(isinstance(frames_count, int))
        assert(frames_count >= 0)
        self.__frames_count = frames_count

    @property
    def EntitiesPerContext(self):
        return self.__frames_count
