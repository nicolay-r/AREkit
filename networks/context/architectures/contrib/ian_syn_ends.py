import numpy as np
import tensorflow as tf
from arekit.networks.context.architectures.ian_frames import IANFrames
from arekit.networks.context.configurations.contrib.ian_syn_ends import IANAttitudeSynonymEndsBasedConfig
from arekit.networks.context.sample import InputSample


class IANAttitudeSynonymEndsBased(IANFrames):
    """
    But original could be based on a pair of attitude ends:
        <Syn_Object_1, Syn_Object2, ... , Syn_Subject_1, Syn_Subject_2 ...>
    """

    I_ENDS = u'synonym_ends'

    def __init__(self):
        super(IANAttitudeSynonymEndsBased, self).__init__()

        self.__ends = None

    # region private methods

    # todo. in help method concat (+sort, +order).
    @staticmethod
    def __compose_ends_from_tensors(obj_ind_input, subj_ind_input):
        assert(isinstance(obj_ind_input, tf.Tensor))
        assert(isinstance(subj_ind_input, tf.Tensor))
        concatenated_result = tf.concat((obj_ind_input, subj_ind_input), axis=1)
        print concatenated_result.shape
        return tf.contrib.framework.sort(concatenated_result, direction='DESCENDING')

    # todo. in help method concat (+sort, +order).
    @staticmethod
    def __compose_ends_from_arrays(obj_ind_input, subj_ind_input):
        assert(isinstance(obj_ind_input, list))
        assert(isinstance(subj_ind_input, list))
        concatenated_result = np.concatenate((obj_ind_input, subj_ind_input), axis=1)
        print concatenated_result.shape
        return -np.sort(-concatenated_result)

    # endregion

    # region public methods

    def init_input(self):
        assert(isinstance(self.Config, IANAttitudeSynonymEndsBasedConfig))

        super(IANAttitudeSynonymEndsBased, self).init_input()

        self.__ends = tf.placeholder(dtype=tf.int32,
                                     shape=[self.Config.BatchSize, self.Config.MaxAspectLength],
                                     name=u'ctx_' + self.I_ENDS)

    def update_network_specific_parameters(self):
        self.__ends = self.__compose_ends_from_tensors(
            obj_ind_input=self.get_input_parameter(InputSample.I_SYN_OBJ_INDS),
            subj_ind_input=self.get_input_parameter(InputSample.I_SYN_SUBJ_INDS))

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IANAttitudeSynonymEndsBased, self).create_feed_dict(input=input,
                                                                              data_type=data_type)

        feed_dict[self.__ends] = self.__compose_ends_from_arrays(
            obj_ind_input=input[InputSample.I_SYN_OBJ_INDS],
            subj_ind_input=input[InputSample.I_SYN_SUBJ_INDS])

        return feed_dict

    # endregion

