from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.labels.scaler import BaseLabelScaler


class SerializationData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, label_scaler, neutral_annot, stemmer):
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(neutral_annot, BaseNeutralAnnotator))
        super(SerializationData, self).__init__(stemmer=stemmer)

        self.__label_scaler = label_scaler

        if self.LabelsCount != neutral_annot.LabelsCount:
            raise Exception(u"Label scaler and neutral annotation are incompatible due to differs in labels count!")

        self.__neutral_annot = neutral_annot

    @property
    def LabelsScaler(self):
        """ Declares the amount of labels utilized in experiment. The latter
            is necessary for conversions from int (uint) to Labels and vice versa.
        """
        return self.__label_scaler

    @property
    def LabelsCount(self):
        return self.__label_scaler.LabelsCount

    @property
    def NeutralAnnotator(self):
        """ Provides an instance of neutral annotator that might be utlized
            for neutral attitudes labeling for a specific set of documents,
            declared in a particular experiment (see OpinionOperations).
        """
        return self.__neutral_annot

    @property
    def StringEntityFormatter(self):
        raise NotImplementedError()

    @property
    def FramesCollection(self):
        raise NotImplementedError()

    @property
    def FrameVariantCollection(self):
        raise NotImplementedError()

    @property
    def TermsPerContext(self):
        raise NotImplementedError
