import random

import tensorflow as tf
import numpy as np
import logging

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.networks.training.bags.bag import Bag
from arekit.common.experiment.data_type import DataType
from arekit.networks.nn import NeuralNetwork
from arekit.networks.training.batch.batch import MiniBatch
from arekit.tests.ctx_compile import contexts_supported

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample


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


def create_minibatch(config, label_scaler):
    assert(isinstance(config, DefaultNetworkConfig))
    assert(isinstance(label_scaler, BaseLabelScaler))

    l_min = min([label_scaler.label_to_uint(l) for l in label_scaler.ordered_suppoted_labels()])
    l_max = max([label_scaler.label_to_uint(l) for l in label_scaler.ordered_suppoted_labels()])

    bags = []
    for i in range(config.BagsPerMinibatch):
        uint_label = random.randint(l_min, l_max)
        label = label_scaler.uint_to_label(uint_label)
        bag = Bag(label)
        for j in range(config.BagSize):
            bag.add_sample(InputSample._generate_test(config))
        bags.append(bag)

    return MiniBatch(bags=bags, batch_id=None)


def test_ctx_feed(network, network_config, create_minibatch_func, logger,
                  display_hidden_values=True,
                  display_idp_values=True):
    assert(isinstance(network, NeuralNetwork))
    assert(isinstance(network_config, DefaultNetworkConfig))
    assert(callable(create_minibatch_func))

    label_scaler = ThreeLabelScaler()
    config = init_config(network_config)
    # Init network.
    network.compile(config=config, reset_graph=True)
    minibatch = create_minibatch_func(config=config,
                                      label_scaler=label_scaler)

    network_optimiser = config.Optimiser.minimize(network.Cost)
    with init_session() as sess:
        # Save graph
        writer = tf.summary.FileWriter("output", sess.graph)
        # Init feed dict
        feed_dict = network.create_feed_dict(input=minibatch.to_network_input(label_scaler=label_scaler),
                                             data_type=DataType.Train)

        hidden_list = list(network.iter_hidden_parameters())
        idp_list = list(network.iter_input_dependent_hidden_parameters())

        hidden_names = [name for name, _ in hidden_list]
        idp_names = [name for name, _ in idp_list]

        fetches_hidden = [tensor for _, tensor in hidden_list]
        fetches_idp = [tensor for _, tensor in idp_list]

        fetches_default = [network_optimiser, network.Cost, network.Accuracy]

        # feed
        result = sess.run(fetches=fetches_default + fetches_hidden + fetches_idp,
                          feed_dict=feed_dict)

        # Printing graph
        print result
        writer.close()

        # Show hidden parameters
        hidden_values = result[len(fetches_default):len(fetches_default) + len(fetches_hidden)]
        for i, value in enumerate(hidden_values):
            if display_hidden_values:
                logger.info('Value type: {}'.format(type(value)))
                logger.info('Hidden parameter "{}": {}'.format(hidden_names[i], value))

        # Show idp parameters
        idp = result[len(fetches_default) + len(fetches_hidden):]
        for i, value in enumerate(idp):
            if display_idp_values:
                logger.info('i: {}'.format(i))
                logger.info('IDP: {}'.format(type(value)))
                logger.info('IDP shape: {}'.format(value.shape))
                logger.info('IDP param/value "{}": {}'.format(idp_names[i], value))


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    for cfg, network in contexts_supported():
        logger.debug("Feed to the network: {}".format(type(network)))
        test_ctx_feed(network=network,
                      network_config=cfg,
                      create_minibatch_func=create_minibatch,
                      logger=logger)
