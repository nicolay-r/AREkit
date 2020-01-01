from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANSynonymEndsAndFramesBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.FramesPerContext + 2 * self.SynonymsPerContext
