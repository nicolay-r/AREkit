import tensorflow as tf
import numpy as np
import logging

from arekit.common.labels.base import PositiveLabel
from arekit.networks.context.configurations.base import DefaultNetworkConfig
from arekit.networks.context.sample import InputSample
from arekit.networks.context.training.bags.bag import Bag
from arekit.networks.context.training.batch import MiniBatch
from arekit.networks.data_type import DataType
from arekit.networks.network import NeuralNetwork
from arekit.tests.ctx_compile import contexts_supported


def init_session():
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    return sess


def init_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    config.notify_initialization_completed()
    return config


def create_minibatch(config):
    assert(isinstance(config, DefaultNetworkConfig))

    bags = []
    label = PositiveLabel()
    empty_sample = InputSample.create_empty(config)
    for i in range(config.BagsPerMinibatch):
        bag = Bag(label)
        for j in range(config.BagSize):
            bag.add_sample(empty_sample)
        bags.append(bag)

    return MiniBatch(bags=bags, batch_id=None)


def test_ctx_feed(network, network_config, create_minibatch_func, display_values=True):
    assert(isinstance(network, NeuralNetwork))
    assert(isinstance(network_config, DefaultNetworkConfig))
    assert(callable(create_minibatch_func))

    config = init_config(network_config)
    # Init network.
    network.compile(config=config, reset_graph=True)
    minibatch = create_minibatch_func(config)

    network_optimiser = config.Optimiser.minimize(network.Cost)
    with init_session() as sess:
        # Init feed dict
        feed_dict = network.create_feed_dict(input=minibatch.to_network_input(),
                                             data_type=DataType.Train)

        hidden_list = list(network.iter_hidden_parameters())
        hidden_names = [name for name, _ in hidden_list]
        fetches_hidden = [tensor for _, tensor in hidden_list]
        fetches_default = [network_optimiser, network.Cost, network.Accuracy]

        # feed
        result = sess.run(fetches=fetches_default + fetches_hidden,
                          feed_dict=feed_dict)

        # Show hidden parameters
        hidden_values = result[len(fetches_default):]
        for i, value in enumerate(hidden_values):
            if display_values:
                logger.info('Value type: {}'.format(type(value)))
            if display_values:
                logger.info('Hidden parameter "{}": {}'.format(hidden_names[i], value))


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    for cfg, network in contexts_supported():
        logger.debug("Feed to the network: {}".format(type(network)))
        test_ctx_feed(network, cfg, create_minibatch)
