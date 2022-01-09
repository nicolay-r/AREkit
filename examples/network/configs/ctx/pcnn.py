import tensorflow as tf
from arekit.contrib.networks.context.configurations.cnn import CNNConfig


def ctx_pcnn_custom_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())
