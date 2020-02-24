from arekit.networks.context.configurations.cnn import CNNConfig


class AttentionCNNBaseConfig(CNNConfig):

    # region properties

    @property
    def AttentionModel(self):
        raise NotImplementedError()

    # endregion

    def get_attention_parameters(self):
        raise NotImplementedError()

    # region public methods

    def _internal_get_parameters(self):
        parameters = super(AttentionCNNBaseConfig, self)._internal_get_parameters()
        parameters += self.get_attention_parameters()
        return parameters

    # endregion
