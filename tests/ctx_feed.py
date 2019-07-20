import tensorflow as tf
import numpy as np

from core.evaluation.labels import PositiveLabel
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.sample import InputSample
from core.networks.context.training.bags.bag import Bag
from core.networks.context.training.batch import MiniBatch
from core.networks.context.training.data_type import DataType
from core.networks.network import NeuralNetwork
from core.tests.ctx_compile import contexts_supported


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


def test_ctx_feed(network, network_config, create_minibatch_func):
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
            print 'Value type: {}'.format(type(value))
            print 'Hidden parameter "{}": {}'.format(hidden_names[i], value)


if __name__ == "__main__":

    for cfg, network in contexts_supported():
        print "Feed to the network: {}".format(type(network))
        test_ctx_feed(network, cfg, create_minibatch)
