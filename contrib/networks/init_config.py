from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


def initialize_config(classes_count,
                      create_config_func,
                      # TODO. Utilize a single modification function.
                      custom_config_modification_func=None,
                      common_config_modification_func=None):
    """
    TODO. Use a single modification function, as it comes from a weird initialization procedure of
          neural network models.
    """
    assert (isinstance(classes_count, int))
    assert (callable(create_config_func) or create_config_func is None)
    assert (callable(common_config_modification_func) or common_config_modification_func is None)
    assert (callable(custom_config_modification_func) or custom_config_modification_func is None)

    # Initialize config
    config = create_config_func()
    assert (isinstance(config, DefaultNetworkConfig))
    config.modify_classes_count(value=classes_count)

    # Common modification func.
    if common_config_modification_func is not None:
        common_config_modification_func(config=config)

    # Custom (post-common) modification func.
    if custom_config_modification_func is not None:
        custom_config_modification_func(config)

    return config
