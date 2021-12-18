import numpy as np

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


def init_config(config, pos_items_count):
    assert(isinstance(config, DefaultNetworkConfig))
    assert(isinstance(pos_items_count, int))

    config.modify_classes_count(3)
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    config.set_pos_count(pos_items_count)

    # Notify other subscribers that initialization process has been completed.
    config.init_initializers()

    # Init other config dependent parameters.
    config.reinit_config_dependent_parameters()
