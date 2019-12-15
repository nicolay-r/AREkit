from arekit.networks.context.configurations.ian_frames import IANFramesConfig


class IANAttitudeSynonymEndsBasedConfig(IANFramesConfig):

    @property
    def MaxAspectLength(self):
        return self.SynonymsPerContext * 2