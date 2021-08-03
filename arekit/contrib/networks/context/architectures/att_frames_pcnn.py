from arekit.contrib.networks.context.architectures.base.att_pcnn_base import AttentionPCNNBase
from arekit.contrib.networks.sample import InputSample


class AttentionFramesPCNN(AttentionPCNNBase):
    """
    Based on a frames in context.
    """

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
