from arekit.common.labels.base import Label
from arekit.common.labels.pair import LabelPair
from arekit.networks.context.sample import InputSample


class Bag:
    """
    Bag is a list of samples (of related contexts)
    and an attitude label
    """

    def __init__(self, label):
        assert(isinstance(label, LabelPair) or
               isinstance(label, Label))
        self.__samples = []
        self.__label = label

    # region properties

    @property
    def BagLabel(self):
        return self.__label

    @property
    def Samples(self):
        return self.__samples

    # endregion

    # region public methods

    def add_sample(self, sample):
        assert(isinstance(sample, InputSample))
        self.__samples.append(sample)

    # endregion

    # region overriden methods

    def __len__(self):
        return len(self.__samples)

    def __iter__(self):
        for sample in self.__samples:
            yield sample

    # endregion
