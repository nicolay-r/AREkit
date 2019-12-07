from arekit.networks.context.configurations.ian_frames import IANFramesConfig


class IANAttitudeEndsBasedConfig(IANFramesConfig):

    @property
    def MaxAspectLength(self):
        return 2
