import tensorflow as tf


class MultiLayerPerceptronAttentionConfig(object):

    __entities_per_context = 2
    __hidden_size = 10

    @property
    def LayerInitializer(self):
        return tf.random_normal_initializer()

    @property
    def EntitiesPerContext(self):
        return self.__entities_per_context

    @property
    def HiddenSize(self):
        return self.__hidden_size

    def get_parameters(self):
        parameters = [
            ("mlp-attention-2016:layer_initializer", self.LayerInitializer),
            ("mlp-attention-2016:entities_per_context", self.EntitiesPerContext),
            ("mlp-attention-2016:hidden_size", self.HiddenSize)
        ]

        return parameters
