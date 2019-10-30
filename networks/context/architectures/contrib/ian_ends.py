import numpy as np
import tensorflow as tf
from core.networks.context.architectures.ian_frames import IANFrames
from core.networks.context.sample import InputSample


class IANAttitudeEndsBased(IANFrames):
    """
    But original could be based on attitute ends.
    """

    I_ENDS = u'ends'

    def __init__(self):
        super(IANAttitudeEndsBased, self).__init__()

        self.__ends = None

    @property
    def AspectInput(self):
        return self.__ends

    def init_input(self):
        super(IANAttitudeEndsBased, self).init_input()
        self.__ends = tf.placeholder(dtype=tf.int32,
                                     shape=[self.Config.BatchSize, 2],
                                     name=u'ctx_' + self.I_ENDS)

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IANAttitudeEndsBased, self).create_feed_dict(input=input,
                                                                       data_type=data_type)

        _i_obj_ind = np.expand_dims(input[InputSample.I_OBJ_IND], axis=1)
        _i_subj_ind = np.expand_dims(input[InputSample.I_SUBJ_IND], axis=1)
        _ends = np.concatenate((_i_obj_ind, _i_subj_ind), axis=1)

        feed_dict[self.__ends] = _ends

        return feed_dict

