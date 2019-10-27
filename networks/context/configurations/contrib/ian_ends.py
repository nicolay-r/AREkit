from core.networks.context.configurations.ian import IANConfig


class IANAttitudeEndsBasedConfig(IANConfig):

    @property
    def MaxAspectLength(self):
        return 2
