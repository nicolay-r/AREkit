from arekit.networks.context.configurations.base import DefaultNetworkConfig


class CNNConfig(DefaultNetworkConfig):

    __window_size = 3
    __filters_count = 300
    __hidden_size = 300

    # region properties

    @property
    def WindowSize(self):
        return self.__window_size

    @property
    def FiltersCount(self):
        return self.__filters_count

    @property
    def HiddenSize(self):
        return self.__hidden_size

    # endregion

    # region public methods

    def _internal_get_parameters(self):
        parameters = super(CNNConfig, self)._internal_get_parameters()

        parameters += [
            ("cnn:filters_count", self.FiltersCount),
            ("cnn:window_size", self.WindowSize),
            ("cnn:hidden_size", self.HiddenSize),
        ]

        return parameters

    # endregion
