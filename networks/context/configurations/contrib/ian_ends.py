from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANEndsBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return 2
