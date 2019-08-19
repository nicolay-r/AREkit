class MultiLayerPerceptronAttentionConfig(object):

    __entities_per_context = 2
    __hidden_size = 10

    @property
    def EntitiesPerContext(self):
        return self.__entities_per_context

    @property
    def HiddenSize(self):
        return self.__hidden_size

    def get_parameters(self):
        parameters = [
            ("attention-yatian-coling-2016:entities_per_context", self.EntitiesPerContext),
            ("attention-yatian-coling-2016:hidden_size", self.HiddenSize)
        ]

        return parameters
