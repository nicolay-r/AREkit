import tensorflow as tf
from arekit.networks.context.architectures.att_pcnn_base import AttentionPCNNBase
from arekit.networks.context.sample import InputSample


class AttentionAttitudeEndsPCNN(AttentionPCNNBase):
    """
    Attention model based on entity pair ends and
    PCNN architecture for sentence encoding.
    """

    def get_att_input(self):
        return tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                         self.get_input_parameter(InputSample.I_OBJ_IND)],
                        axis=-1)