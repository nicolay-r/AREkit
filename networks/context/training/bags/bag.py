from core.evaluation.labels import LabelPair, Label
from core.networks.context.training.sample import Sample


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

    def add_sample(self, sample):
        assert(isinstance(sample, Sample))
        self.__samples.append(sample)

    @property
    def BagLabel(self):
        return self.__label

    @property
    def Samples(self):
        return self.__samples

    def __len__(self):
        return len(self.__samples)

    def __iter__(self):
        for sample in self.__samples:
            yield sample
