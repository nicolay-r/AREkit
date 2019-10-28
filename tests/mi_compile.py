#!/usr/bin/python
from core.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from core.networks.multi.configuration.base import BaseMultiInstanceConfig
from core.tests.ctx_compile import init_config
from core.tests.ctx_feed import contexts_supported


def test_mpmi(context_config, context_network):
    print "TEST: {}".format(context_network)
    config = BaseMultiInstanceConfig(context_config)
    # TODO. Provide other examples.
    network = MaxPoolingOverSentences(context_network=context_network)
    init_config(config)
    network.compile(config, reset_graph=True)


if __name__ == "__main__":

    for config, network in contexts_supported():
        test_mpmi(config, network)
