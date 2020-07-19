import logging
import sys
import unittest


sys.path.append('../../')

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.base import PositiveLabel

from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences

from arekit.networks.feeding.bags.bag import Bag
from arekit.networks.feeding.batch.multi import MultiInstanceMiniBatch

from arekit.tests.test_tf_ctx_feed import TestContextNetworkFeeding
from arekit.tests.tf_networks.supported import get_supported


class TestMultiInstanceFeed(unittest.TestCase):

    @staticmethod
    def __create_minibatch(config, labels_scaler):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(labels_scaler, BaseLabelScaler))
        bags = []
        label = PositiveLabel()
        empty_sample = InputSample.create_empty(config)
        for i in range(config.BagsPerMinibatch):
            bag = Bag(label)
            for j in range(config.BagSize):
                bag.add_sample(empty_sample)
            bags.append(bag)

        return MultiInstanceMiniBatch(bags=bags, batch_id=None)

    @staticmethod
    def multiinstances_supported(ctx_config, ctx_network):
        return [
            (MaxPoolingOverSentencesConfig(ctx_config), MaxPoolingOverSentences(ctx_network)),
            # (AttSelfOverSentencesConfig(ctx_config), AttSelfOverSentences(ctx_network))
        ]

    def test(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        for ctx_config, ctx_network in get_supported():
            for config, network in self.multiinstances_supported(ctx_config, ctx_network):
                logger.info(type(network))
                logger.info(u'\t-> {}'.format(type(ctx_network)))
                TestContextNetworkFeeding.run_feeding(network=network,
                                                      network_config=config,
                                                      create_minibatch_func=self.__create_minibatch,
                                                      logger=logger,
                                                      display_idp_values=False)


if __name__ == '__main__':
    unittest.main()
