from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.opinions.annot.base import BaseOpinionAnnotator


class ExperimentSerializationContext(ExperimentContext):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, label_scaler, annot, name_provider, data_folding):
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(annot, BaseOpinionAnnotator) or annot is None)
        super(ExperimentSerializationContext, self).__init__(name_provider=name_provider,
                                                             data_folding=data_folding)

        self.__label_scaler = label_scaler

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
    def FramesConnotationProvider(self):
        raise NotImplementedError()

    @property
    def FrameVariantCollection(self):
        raise NotImplementedError()

    @property
    def TermsPerContext(self):
        raise NotImplementedError
