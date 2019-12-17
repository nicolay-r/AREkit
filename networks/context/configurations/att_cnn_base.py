from arekit.networks.context.configurations.cnn import CNNConfig


class AttentionCNNBaseConfig(CNNConfig):

    # region properties

    @property
    def AttentionModel(self):
        raise NotImplementedError()

    @property
    def MLPAttentionConfig(self):
        raise NotImplementedError()

    # endregion

    # region public methods

    def _internal_get_parameters(self):
        parameters = super(AttentionCNNBaseConfig, self)._internal_get_parameters()
        parameters += self.MLPAttentionConfig.get_parameters()
        return parameters

    # endregion
