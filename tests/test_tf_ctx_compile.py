#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import sys
import unittest

sys.path.append('../../')


from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.tests.tf_networks.utils import init_config
from arekit.tests.tf_networks.supported import get_supported


class TestContextNetworkCompilation(unittest.TestCase):

    def test(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        for config, network in get_supported():
            assert(isinstance(config, DefaultNetworkConfig))
            config.modify_classes_count(3)

            logger.info("Compile: {}".format(type(network)))
            logger.info("Clases count: {}".format(config.ClassesCount))

            init_config(config)
            network.compile(config, reset_graph=True)


if __name__ == '__main__':
    unittest.main()
