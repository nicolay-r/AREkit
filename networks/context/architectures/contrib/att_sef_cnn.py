import tensorflow as tf
from arekit.networks.context.architectures.att_cnn_base import AttentionCNNBase
from arekit.networks.context.sample import InputSample


class AttentionSynonymEndsAndFramesCNN(AttentionCNNBase):

    def get_att_input(self):
        combined = tf.concat([self.get_input_parameter(InputSample.I_SYN_SUBJ_INDS),
                              self.get_input_parameter(InputSample.I_SYN_OBJ_INDS),
                              self.get_input_parameter(InputSample.I_FRAME_INDS)],
                             axis=-1)
        return tf.contrib.framework.sort(combined, direction='DESCENDING')

