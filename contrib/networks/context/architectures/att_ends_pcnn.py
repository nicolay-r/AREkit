import tensorflow as tf
from arekit.contrib.networks.context.architectures.base.att_pcnn_base import AttentionPCNNBase
from arekit.contrib.networks.sample import InputSample


class AttentionEndsPCNN(AttentionPCNNBase):
    """
    Attention model based on entity pair ends and
    PCNN architecture for sentence encoding.
    """

    def get_att_input(self):
        return tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                         self.get_input_parameter(InputSample.I_OBJ_IND)],
                        axis=-1)
