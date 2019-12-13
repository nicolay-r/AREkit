import numpy as np
import tensorflow as tf
from arekit.networks.context.architectures.ian_frames import IANFrames
from arekit.networks.context.configurations.contrib.ian_ends import IANAttitudeEndsBasedConfig
from arekit.networks.context.sample import InputSample


class IANAttitudeEndsBased(IANFrames):
    """
    But original could be based on attitude ends.
    """

    I_ENDS = u'ends'

    def __init__(self):
        super(IANAttitudeEndsBased, self).__init__()

        self.__ends = None

    @property
    def AspectInput(self):
        return self.__ends

    def init_input(self):
        assert(isinstance(self.Config, IANAttitudeEndsBasedConfig))
        super(IANAttitudeEndsBased, self).init_input()
        self.__ends = tf.placeholder(dtype=tf.int32,
                                     shape=[self.Config.BatchSize, self.Config.MaxAspectLength],
                                     name=u'ctx_' + self.I_ENDS)

    @staticmethod
    def __compose_opinion_ends_from_tensors(obj_ind_input, subj_ind_input):
        assert(isinstance(obj_ind_input, tf.Tensor))
        assert(isinstance(subj_ind_input, tf.Tensor))
        i_obj_ind = tf.expand_dims(obj_ind_input, axis=1)
        i_subj_ind = tf.expand_dims(subj_ind_input, axis=1)
        return tf.concat((i_obj_ind, i_subj_ind), axis=1)

    @staticmethod
    def __compose_opinion_ends_from_arrays(obj_ind_input, subj_ind_input):
        assert(isinstance(obj_ind_input, list))
        assert(isinstance(subj_ind_input, list))
        i_obj_ind = np.expand_dims(obj_ind_input, axis=1)
        i_subj_ind = np.expand_dims(subj_ind_input, axis=1)
        return np.concatenate((i_obj_ind, i_subj_ind), axis=1)

    def update_network_specific_parameters(self):
        self.__ends = self.__compose_opinion_ends_from_tensors(
            obj_ind_input=self.get_input_parameter(InputSample.I_OBJ_IND),
            subj_ind_input=self.get_input_parameter(InputSample.I_SUBJ_IND))

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IANAttitudeEndsBased, self).create_feed_dict(input=input,
                                                                       data_type=data_type)
        feed_dict[self.__ends] = self.__compose_opinion_ends_from_arrays(
            obj_ind_input=input[InputSample.I_OBJ_IND],
            subj_ind_input=input[InputSample.I_SUBJ_IND])

        return feed_dict
