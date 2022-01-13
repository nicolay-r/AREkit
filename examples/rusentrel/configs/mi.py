from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from examples.rusentrel.configs.common import MI_CONTEXTS_PER_OPINION


def apply_ds_mi_settings(config):
    """
    This function describes a base config setup for all models.
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(2)