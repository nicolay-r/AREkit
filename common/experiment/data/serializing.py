from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.neutral.annot.factory import create_annotator


class SerializationData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, labels_scaler, stemmer):
        super(SerializationData, self).__init__(labels_scaler=labels_scaler,
                                                stemmer=stemmer)

        self.__neutral_annot = create_annotator(
            labels_count=labels_scaler.LabelsCount,
            dist_in_terms_between_opin_ends=self.DistanceInTermsBetweenOpinionEndsBound)

    @property
    def NeutralAnnotator(self):
        """ Provides an instance of neutral annotator that might be utlized
            for neutral attitudes labeling for a specific set of documents,
            declared in a particular experiment (see OpinionOperations).
        """
        return self.__neutral_annot

    @property
    def DistanceInTermsBetweenOpinionEndsBound(self):
        raise NotImplementedError()

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
