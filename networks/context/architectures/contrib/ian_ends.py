import numpy as np
import tensorflow as tf
from core.networks.context.architectures.ian import IAN
from core.networks.context.sample import InputSample


class IANAttituteEndsBased(IAN):
    """
    But original could be based on attitute ends.
    """

    I_ENDS = u'ends'

    def __init__(self):
        super(IANAttituteEndsBased, self).__init__()

    @property
    def AspectInput(self):
        return self.__ends

    def init_input(self):
        super(IANAttituteEndsBased, self).init_input()
        self.__ends = tf.placeholder(dtype=tf.int32,
                                     shape=[self.Config.BatchSize, 2],
                                     name=u'ctx_' + u'_ends')

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IANAttituteEndsBased, self).create_feed_dict(input=input,
                                                                       data_type=data_type)

        feed_dict[self.__ends] = np.concatenate((input[InputSample.I_OBJ_IND],
                                                 input[InputSample.I_SUBJ_IND]),
                                                axis=1)

