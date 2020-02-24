from arekit.contrib.networks.context.architectures.base.att_bilstm_base import AttentionBiLSTMBase
from arekit.networks.context.sample import InputSample


class AttentionFramesBiLSTM(AttentionBiLSTMBase):
    """
    Based on a frames in context.
    """

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
