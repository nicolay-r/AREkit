from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANFramesConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.FramesPerContext
