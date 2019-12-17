from arekit.networks.context.architectures.att_pcnn_base import AttentionPCNNBase
from arekit.networks.context.sample import InputSample


class AttentionFramesPCNN(AttentionPCNNBase):
    """
    Based on a frames in context.
    """

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
