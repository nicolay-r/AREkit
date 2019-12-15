#!/usr/bin/python
import logging
from arekit.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.networks.multi.configuration.base import BaseMultiInstanceConfig
from arekit.tests.ctx_compile import init_config
from arekit.tests.ctx_feed import contexts_supported


def test_mpmi(context_config, context_network):
    logging.info("TEST: {}".format(context_network))
    config = BaseMultiInstanceConfig(context_config)
    # TODO. Provide other examples.
    network = MaxPoolingOverSentences(context_network=context_network)
    init_config(config)
    network.compile(config, reset_graph=True)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    for config, network in contexts_supported():
        test_mpmi(config, network)
