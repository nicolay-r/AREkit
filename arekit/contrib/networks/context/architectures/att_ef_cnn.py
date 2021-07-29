import tensorflow as tf
from arekit.contrib.networks.context.architectures.base.att_cnn_base import AttentionCNNBase
from arekit.contrib.networks.sample import InputSample


class AttentionEndsAndFramesCNN(AttentionCNNBase):

    def get_att_input(self):
        combined = tf.concat([
            self.get_input_parameter(InputSample.I_FRAME_INDS),
            tf.expand_dims(self.get_input_parameter(InputSample.I_SUBJ_IND), axis=-1),
            tf.expand_dims(self.get_input_parameter(InputSample.I_OBJ_IND), axis=-1)],
            axis=-1)
        return tf.contrib.framework.sort(combined, direction='DESCENDING')
