from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANAttitudeEndsBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return 2
