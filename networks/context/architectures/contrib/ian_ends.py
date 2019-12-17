import tensorflow as tf
from arekit.networks.context.architectures.ian_base import IANBase
from arekit.networks.context.sample import InputSample


class IANAttitudeEndsBased(IANBase):
    """
    Based on a pair of attitude ends: <Object, Subject>
    """

    def get_aspect_input(self):
        return tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                         self.get_input_parameter(InputSample.I_OBJ_IND)],
                        axis=-1)
