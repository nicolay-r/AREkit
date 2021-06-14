from arekit.contrib.networks.sample import InputSample


class Bag:
    """
    Bag is a list of samples (of related contexts)
    and an attitude label
    """

    def __init__(self, uint_label):
        assert(isinstance(uint_label, int) or uint_label is None)
        self.__samples = []
        self.__uint_label = uint_label

    # region properties

    @property
    def UintBagLabel(self):
        return self.__uint_label

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
