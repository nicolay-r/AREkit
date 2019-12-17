from arekit.networks.context.configurations.ian_base import IANBaseConfig


class IANAttitudeSynonymEndsBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.SynonymsPerContext * 2