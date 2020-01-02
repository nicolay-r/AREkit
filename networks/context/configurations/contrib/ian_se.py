from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANSynonymEndsBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.SynonymsPerContext * 2