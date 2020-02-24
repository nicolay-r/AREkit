import tensorflow as tf
from arekit.contrib.networks.context.architectures.base.att_cnn_base import AttentionCNNBase
from arekit.networks.context.sample import InputSample


class AttentionSynonymEndsCNN(AttentionCNNBase):
    """
    But original could be based on a pair of attitude ends:
        <Syn_Object_1, Syn_Object2, ... , Syn_Subject_1, Syn_Subject_2 ...>
    """

    def get_att_input(self):
        combined = tf.concat([self.get_input_parameter(InputSample.I_SYN_SUBJ_INDS),
                              self.get_input_parameter(InputSample.I_SYN_OBJ_INDS)],
                             axis=-1)
        return tf.contrib.framework.sort(combined, direction='DESCENDING')

