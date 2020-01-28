import tensorflow as tf
from arekit.networks.attention import common
from arekit.networks.context.architectures.rcnn import RCNN


class AttentionSelfRCNN(RCNN):

    def __init__(self):
        super(AttentionSelfRCNN, self).__init__()
        self.__att_alphas = None

    def get_attention_alphas(self, rnn_outputs):
        raise NotImplementedError()

    # region public methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionSelfRCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion

    def modify_rnn_outputs_optional(self, output_fw, output_bw):
        rnn_outputs = tf.add(output_fw, output_bw)

        self.__att_alphas = self.get_attention_alphas(rnn_outputs)

        output_fw_w = output_fw * tf.expand_dims(self.__att_alphas, -1)
        output_bw_w = output_fw * tf.expand_dims(self.__att_alphas, -1)

        return output_fw_w, output_bw_w
