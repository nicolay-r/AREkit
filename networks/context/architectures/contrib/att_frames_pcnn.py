from arekit.networks.context.architectures.att_ends_pcnn import AttentionAttitudeEndsPCNN
from arekit.networks.context.sample import InputSample


class AttentionFramesPCNN(AttentionAttitudeEndsPCNN):

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
