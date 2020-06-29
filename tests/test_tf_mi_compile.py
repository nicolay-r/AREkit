#!/usr/bin/python
import logging
import sys
import unittest

sys.path.append('../')


from arekit.tests.tf_networks.supported import get_supported
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from arekit.tests.tf_networks.utils import init_config


class TestMultiInstanceCompile(unittest.TestCase):

    @staticmethod
    def mpmi(context_config, context_network):
        assert(isinstance(context_config, DefaultNetworkConfig))

        context_config.modify_classes_count(3)

        logging.info("TEST: {}".format(context_network))
        config = BaseMultiInstanceConfig(context_config)

        config.modify_classes_count(3)

        network = MaxPoolingOverSentences(context_network=context_network)
        init_config(config)
        network.compile(config, reset_graph=True)

    def test(self):
        logging.basicConfig(level=logging.INFO)

        for config, network in get_supported():
            self.mpmi(config, network)


if __name__ == '__main__':
    unittest.main()
