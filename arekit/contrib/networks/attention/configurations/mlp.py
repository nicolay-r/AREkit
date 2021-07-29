import tensorflow as tf


class MLPAttentionConfig(object):

    __keys_per_context = 2
    __hidden_size = 10

    # region properties

    @property
    def LayerInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    @property
    def KeysPerContext(self):
        return self.__keys_per_context

    @property
    def HiddenSize(self):
        return self.__hidden_size

    # endregion

    # region public methods

    def get_parameters(self):
        parameters = [
            ("mlp-attention-2016:layer_initializer", self.LayerInitializer),
            ("mlp-attention-2016:keys_per_context", self.KeysPerContext),
            ("mlp-attention-2016:hidden_size", self.HiddenSize)
        ]

        return parameters

    # endregion
