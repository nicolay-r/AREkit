from arekit.networks.context.architectures.ian_base import IANBase
from arekit.networks.context.sample import InputSample


class IANFrames(IANBase):
    """
    Based on a frames in context.
    """

    def get_aspect_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)