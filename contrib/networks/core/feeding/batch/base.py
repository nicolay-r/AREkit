import logging
from collections import OrderedDict

from arekit.common.model.sample import InputSampleBase
from arekit.contrib.networks.core.debug import DebugKeys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MiniBatch(object):
    """
    Is a neural network batch that is consist of bags.
    """

    I_LABELS = 'y'

    def __init__(self, bags, batch_id=None):
        assert(isinstance(batch_id, int) or batch_id is None)
        assert(isinstance(bags, list))
        self.__batch_id = batch_id
        self.__bags = bags

    # region properties

    @property
    def Bags(self):
        return self.__bags

    # endregion

    # region public methods

    def to_network_input(self, provide_labels):

        result = OrderedDict()

        for sample in self.iter_by_samples():

            assert(isinstance(sample, InputSampleBase))

            for arg, value in sample:
                if arg not in result:
                    result[arg] = []
                result[arg].append(value)

        for bag in self.iter_by_bags():
            if self.I_LABELS not in result:
                result[self.I_LABELS] = []
            uint_label = bag.UintBagLabel if provide_labels else 0
            result[self.I_LABELS].append(uint_label)

        if DebugKeys.MiniBatchShow:
            MiniBatch.debug_output(result)

        return result

    # endregion

    # region public 'debug' methods

    @staticmethod
    def debug_output(result):
        logger.debug("-------------------")
        for key, value in result.iteritems():
            logger.debug("{}: {}".format(key, value))
        logger.debug("-------------------")

    # endregion

    # region public 'iter' methods

    def iter_by_samples(self):
        for bag in self.__bags:
            for sample in bag:
                yield sample

    def iter_by_bags(self):
        for bag in self.__bags:
            yield bag

    # endregion
