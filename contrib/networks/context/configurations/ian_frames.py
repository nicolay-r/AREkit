from arekit.contrib.networks.context.configurations.base.ian_base import IANBaseConfig


class IANFramesConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.FramesPerContext
