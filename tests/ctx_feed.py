import tensorflow as tf
import numpy as np

from core.evaluation.labels import PositiveLabel
from core.networks.context.architectures.att_cnn import AttentionCNN
from core.networks.context.architectures.bi_lstm import BiLSTM
from core.networks.context.architectures.cnn import VanillaCNN
from core.networks.context.architectures.ian import IAN
from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.architectures.rcnn import RCNN
from core.networks.context.architectures.rnn import RNN
from core.networks.context.configurations.att_cnn import AttentionCNNConfig
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.configurations.cnn import CNNConfig
from core.networks.context.configurations.ian import IANConfig
from core.networks.context.configurations.rcnn import RCNNConfig
from core.networks.context.configurations.rnn import RNNConfig
from core.networks.context.sample import InputSample
from core.networks.context.training.bags.bag import Bag
from core.networks.context.training.batch import MiniBatch
from core.networks.context.training.data_type import DataType


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


def test_feed(network, network_config):
    config = init_config(network_config)
    # Init network.
    network.compile(config=config, reset_graph=True)
    minibatch = create_minibatch(config)

    network_optimiser = config.Optimiser.minimize(network.Cost)
    with init_session() as sess:
        # Init feed dict
        feed_dict = network.create_feed_dict(input=minibatch.to_network_input(),
                                             data_type=DataType.Train)
        # feed
        result = sess.run(fetches=[network_optimiser,
                                   network.Cost,
                                   network.Accuracy],
                          feed_dict=feed_dict)

        print result


test_feed(network_config=CNNConfig(), network=VanillaCNN())
test_feed(network_config=CNNConfig(), network=PiecewiseCNN())
test_feed(network_config=AttentionCNNConfig(), network=AttentionCNN())
# test_feed(network_config=RNNConfig(), network=RNN())
test_feed(network_config=BiLSTMConfig(), network=BiLSTM())
test_feed(network_config=RCNNConfig(), network=RCNN())
test_feed(network_config=IANConfig(), network=IAN())
