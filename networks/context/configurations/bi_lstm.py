from base import DefaultNetworkConfig


class BiLSTMConfig(DefaultNetworkConfig):

    __hidden_size = 128

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    def LearningRate(self):
        return 0.1

    def modify_hidden_size(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__hidden_size = value

    def _internal_get_parameters(self):
        parameters = super(BiLSTMConfig, self)._internal_get_parameters()

        parameters += [
            ("bi-lstm:hidden_size", self.HiddenSize)
        ]

        return parameters

