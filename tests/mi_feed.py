from core.evaluation.labels import PositiveLabel
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.sample import InputSample
from core.networks.context.training.bags.bag import Bag
from core.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from core.networks.multi.configuration.base import BaseMultiInstanceConfig
from core.networks.multi.training.batch import MultiInstanceBatch
from ctx_feed import test_ctx_feed, contexts_supported


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

    return MultiInstanceBatch(bags=bags, batch_id=None)


if __name__ == "__main__":

    for ctx_config, ctx_network in contexts_supported():
        config = BaseMultiInstanceConfig(ctx_config)
        network = MaxPoolingOverSentences(ctx_network)
        print type(ctx_network)
        test_ctx_feed(network=network,
                      network_config=config,
                      create_minibatch_func=create_minibatch)
