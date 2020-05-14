import logging
import ctx_feed
from arekit.common.labels.base import PositiveLabel
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample
from arekit.networks.training.bags.bag import Bag
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.networks.training.batch.multi.batch import MultiInstanceBatch


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
        # (AttSelfOverSentencesConfig(ctx_config), AttSelfOverSentences(ctx_network))
    ]


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    for ctx_config, ctx_network in ctx_feed.contexts_supported():
        for config, network in multiinstances_supported(ctx_config, ctx_network):
            logger.info(type(network))
            logger.info(u'\t-> {}'.format(type(ctx_network)))
            ctx_feed.test_ctx_feed(network=network,
                                   network_config=config,
                                   create_minibatch_func=create_minibatch,
                                   logger=logger,
                                   display_idp_values=False)
