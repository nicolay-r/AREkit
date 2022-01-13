from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from examples.rusentrel.configs.common import apply_classic_mi_settings
from examples.rusentrel.configs.mi import apply_ds_mi_settings
from examples.rusentrel.configs.single import ctx_self_att_bilstm_custom_config, ctx_att_bilstm_p_zhou_custom_config, \
    ctx_att_bilstm_z_yang_custom_config, ctx_bilstm_custom_config, ctx_cnn_custom_config, ctx_lstm_custom_config, \
    ctx_pcnn_custom_config, ctx_rcnn_custom_config, ctx_rcnn_z_yang_custom_config, ctx_rcnn_p_zhou_custom_config


def modify_config_for_model(model_name, model_input_type, config):
    assert(isinstance(model_name, ModelNames))
    assert(isinstance(model_input_type, ModelInputType))
    assert (isinstance(config, DefaultNetworkConfig))

    if model_input_type == ModelInputType.SingleInstance:
        if model_name == ModelNames.SelfAttentionBiLSTM:
            ctx_self_att_bilstm_custom_config(config)
        if model_name == ModelNames.AttSelfPZhouBiLSTM:
            ctx_att_bilstm_p_zhou_custom_config(config)
        if model_name == ModelNames.AttSelfZYangBiLSTM:
            ctx_att_bilstm_z_yang_custom_config(config)
        if model_name == ModelNames.BiLSTM:
            ctx_bilstm_custom_config(config)
        if model_name == ModelNames.CNN:
            ctx_cnn_custom_config(config)
        if model_name == ModelNames.LSTM:
            ctx_lstm_custom_config(config)
        if model_name == ModelNames.PCNN:
            ctx_pcnn_custom_config(config)
        if model_name == ModelNames.RCNN:
            ctx_rcnn_custom_config(config)
        if model_name == ModelNames.RCNNAttZYang:
            ctx_rcnn_z_yang_custom_config(config)
        if model_name == ModelNames.RCNNAttPZhou:
            ctx_rcnn_p_zhou_custom_config(config)

        return

    if model_input_type == ModelInputType.MultiInstanceMaxPooling or \
       model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        assert(isinstance(config, BaseMultiInstanceConfig))

        # We assign all the settings related to the case of
        # single instance model, for the related ContextConfig.
        modify_config_for_model(model_name=model_name,
                                model_input_type=ModelInputType.SingleInstance,
                                config=config.ContextConfig)

        # We apply modification of some parameters
        config.fix_context_parameters()
        return

    raise NotImplementedError(u"Model input type {input_type} is not supported".format(
        input_type=model_input_type))


def optionally_modify_config_for_experiment(config, exp_type, model_input_type):
    assert(isinstance(exp_type, ExperimentTypes))
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        if exp_type == ExperimentTypes.RuSentRel:
            apply_classic_mi_settings(config)
        if exp_type == ExperimentTypes.RuAttitudes or exp_type == ExperimentTypes.RuSentRelWithRuAttitudes:
            apply_ds_mi_settings(config)

        return
