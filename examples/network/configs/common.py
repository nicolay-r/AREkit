from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from examples.network.args.const import BAGS_PER_MINIBATCH

MI_CONTEXTS_PER_OPINION = 3


def apply_classic_mi_settings(config):
    """
    Multi instance version
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(BAGS_PER_MINIBATCH)
