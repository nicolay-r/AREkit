class AttentionConfig(object):

    __entities_per_context = 2
    __hidden_size = 10
    # TODO. Dropout is not used
    __dropout_keep_prob = 0.9        # Not used now actually

    @property
    def EntitiesPerContext(self):
        return self.__entities_per_context

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    # TODO. Dropout is not used
    def DropoutKeepProb(self):
        return self.__dropout_keep_prob

    def get_parameters(self):
        parameters = [
            ("attention:entities_per_context", self.EntitiesPerContext),
            ("attention:hidden_size", self.HiddenSize),
            ("attention:dropout_keep_prob", self.DropoutKeepProb),
        ]

        return parameters
