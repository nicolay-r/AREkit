from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANEndsAndFramesConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.FramesPerContext + 2
