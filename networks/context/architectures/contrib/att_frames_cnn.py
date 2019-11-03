from core.networks.context.architectures.att_ends_cnn import AttentionCNN
from core.networks.context.sample import InputSample


class AttentionFramesCNN(AttentionCNN):

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
