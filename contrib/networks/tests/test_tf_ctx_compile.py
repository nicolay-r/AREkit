import logging
import sys
import unittest

sys.path.append('../../../')

from arekit.common.languages.ru.pos_service import PartOfSpeechTypesService
from arekit.contrib.networks.tests.tf_networks.utils import init_config
from arekit.contrib.networks.tests.tf_networks.supported import get_supported
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


class TestContextNetworkCompilation(unittest.TestCase):

    def test(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        for config, network in get_supported():
            assert(isinstance(config, DefaultNetworkConfig))
            config.modify_classes_count(3)
            config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())

            logger.info("Compile: {}".format(type(network)))
            logger.info("Clases count: {}".format(config.ClassesCount))

            init_config(config)
            network.compile(config, reset_graph=True, graph_seed=42)


if __name__ == '__main__':
    unittest.main()
