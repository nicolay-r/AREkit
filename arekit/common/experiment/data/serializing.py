from arekit.common.experiment.annot.base import BaseAnnotator
from arekit.common.experiment.data.base import DataIO
from arekit.common.labels.scaler import BaseLabelScaler


class SerializationData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, label_scaler, annot, stemmer):
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(annot, BaseAnnotator))
        super(SerializationData, self).__init__(stemmer=stemmer)

        self.__label_scaler = label_scaler

        if self.LabelsCount != annot.LabelsCount:
            raise Exception("Label scaler and annotator are incompatible due to differs in labels count!")

        self.__annot = annot

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
    def Annotator(self):
        """ Provides an instance of annotator that might be utilized
            for attitudes labeling within a specific set of documents,
            declared in a particular experiment (see OpinionOperations).
        """
        return self.__annot

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
