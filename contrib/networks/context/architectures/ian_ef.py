import tensorflow as tf
from arekit.contrib.networks.context.architectures.base.ian_base import IANBase
from arekit.networks.context.sample import InputSample


class IANEndsAndFrames(IANBase):

    def get_aspect_input(self):
        combined = tf.concat([
            self.get_input_parameter(InputSample.I_FRAME_INDS),
            tf.expand_dims(self.get_input_parameter(InputSample.I_SUBJ_IND), axis=-1),
            tf.expand_dims(self.get_input_parameter(InputSample.I_OBJ_IND), axis=-1)],
            axis=-1)
        return tf.contrib.framework.sort(combined, direction='DESCENDING')

