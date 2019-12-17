import numpy as np
import tensorflow as tf
from arekit.networks.context.architectures.contrib.ian_frames import IANFrames
from arekit.networks.context.configurations.contrib.ian_syn_ends import IANAttitudeSynonymEndsBasedConfig
from arekit.networks.context.sample import InputSample


class IANAttitudeSynonymEndsBased(IANFrames):
    """
    But original could be based on a pair of attitude ends:
        <Syn_Object_1, Syn_Object2, ... , Syn_Subject_1, Syn_Subject_2 ...>
    """

    def get_aspect_input(self):
        # TODO. There is a still bug during feed operation.
        combined = tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                             self.get_input_parameter(InputSample.I_OBJ_IND)],
                            axis=-1)

        return tf.contrib.framework.sort(combined, direction='DESCENDING')

