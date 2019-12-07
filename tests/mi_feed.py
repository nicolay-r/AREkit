from arekit.common.labels.base import PositiveLabel
from arekit.networks.context.configurations.base import DefaultNetworkConfig
from arekit.networks.context.sample import InputSample
from arekit.networks.context.training.bags.bag import Bag
from arekit.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.networks.multi.configuration.max_pooling import MaxPoolingOverSentencesConfig
from arekit.networks.multi.training.batch import MultiInstanceBatch
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


def multiinstances_supported(ctx_config, ctx_network):
    return [
        (MaxPoolingOverSentencesConfig(ctx_config), MaxPoolingOverSentences(ctx_network)),
        # (AttHiddenOverSentencesConfig(ctx_config), AttHiddenOverSentences(ctx_network))
    ]


if __name__ == "__main__":

    for ctx_config, ctx_network in contexts_supported():
        for config, network in multiinstances_supported(ctx_config, ctx_network):
            print type(network)
            print u'\t-> {}'.format(type(ctx_network))
            test_ctx_feed(network=network,
                          network_config=config,
                          create_minibatch_func=create_minibatch,
                          display_values=False)
