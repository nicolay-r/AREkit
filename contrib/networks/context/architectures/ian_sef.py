import tensorflow as tf
from arekit.contrib.networks.context.architectures.base.ian_base import IANBase
from arekit.networks.context.sample import InputSample


class IANSynonymEndsAndFrames(IANBase):

    def get_aspect_input(self):
        combined = tf.concat([self.get_input_parameter(InputSample.I_SYN_SUBJ_INDS),
                              self.get_input_parameter(InputSample.I_SYN_OBJ_INDS),
                              self.get_input_parameter(InputSample.I_FRAME_INDS)],
                             axis=-1)
        return tf.contrib.framework.sort(combined, direction='DESCENDING')
