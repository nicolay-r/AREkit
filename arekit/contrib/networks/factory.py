from arekit.contrib.networks.context.architectures.att_ends_cnn import AttentionEndsCNN
from arekit.contrib.networks.context.architectures.att_ends_pcnn import AttentionEndsPCNN
from arekit.contrib.networks.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arekit.contrib.networks.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.contrib.networks.context.architectures.cnn import VanillaCNN
from arekit.contrib.networks.context.architectures.ian_ends import IANEndsBased
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.architectures.rcnn import RCNN
from arekit.contrib.networks.context.architectures.rnn import RNN
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.att_ends_cnn import AttentionEndsCNNConfig
from arekit.contrib.networks.context.configurations.att_ends_pcnn import AttentionEndsPCNNConfig
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.context.configurations.ian_ends import IANEndsBasedConfig
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from arekit.contrib.networks.multi.architectures.base.base import BaseMultiInstanceNeuralNetwork
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig


def create_network_and_network_config_funcs(model_name, model_input_type):
    assert(isinstance(model_name, ModelNames))
    assert(isinstance(model_input_type, ModelInputType))

    ctx_network_func, ctx_config_func = __get_network_with_config_types(model_name)

    if model_input_type == ModelInputType.SingleInstance:
        # In case of a single instance model, there is no need to perform extra wrapping
        # since all the base models assumes to work with a single context (input).
        return ctx_network_func, ctx_config_func

    # Compose multi-instance neural network and related configuration
    # in a form of a wrapper over context-based neural network and configuration respectively.
    mi_network, mi_config = __get_mi_network_with_config(model_input_type)
    assert(issubclass(mi_network, BaseMultiInstanceNeuralNetwork))
    assert(issubclass(mi_config, BaseMultiInstanceConfig))
    return lambda: mi_network(context_network=ctx_network_func()), \
           lambda: mi_config(context_config=ctx_config_func())


def __get_mi_network_with_config(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))
    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MaxPoolingOverSentences, MaxPoolingOverSentencesConfig
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return AttSelfOverSentences, AttSelfOverSentencesConfig


def __get_network_with_config_types(model_name):
    assert(isinstance(model_name, ModelNames))
    if model_name == ModelNames.SelfAttentionBiLSTM:
        return SelfAttentionBiLSTM, SelfAttentionBiLSTMConfig
    if model_name == ModelNames.AttSelfPZhouBiLSTM:
        return AttentionSelfPZhouBiLSTM, AttentionSelfPZhouBiLSTMConfig
    if model_name == ModelNames.AttSelfZYangBiLSTM:
        return AttentionSelfZYangBiLSTM, AttentionSelfZYangBiLSTMConfig
    if model_name == ModelNames.BiLSTM:
        return BiLSTM, BiLSTMConfig
    if model_name == ModelNames.CNN:
        return VanillaCNN, CNNConfig
    if model_name == ModelNames.LSTM:
        return RNN, RNNConfig
    if model_name == ModelNames.PCNN:
        return PiecewiseCNN, CNNConfig
    if model_name == ModelNames.RCNN:
        return RCNN, RCNNConfig
    if model_name == ModelNames.RCNNAttZYang:
        return AttentionSelfZYangRCNN, RCNNConfig
    if model_name == ModelNames.RCNNAttPZhou:
        return AttentionSelfPZhouRCNN, RCNNConfig
    if model_name == ModelNames.IANEnds:
        return IANEndsBased, IANEndsBasedConfig
    if model_name == ModelNames.AttEndsPCNN:
        return AttentionEndsPCNN, AttentionEndsPCNNConfig
    if model_name == ModelNames.AttEndsCNN:
        return AttentionEndsCNN, AttentionEndsCNNConfig
    raise NotImplementedError(u"config was not implemented for `{}` model name".format(model_name))
