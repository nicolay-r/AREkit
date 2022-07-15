from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.labels.scaler.base import BaseLabelScaler


class ExperimentSerializationContext(ExperimentContext):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, label_scaler, name_provider):
        assert(isinstance(label_scaler, BaseLabelScaler))
        super(ExperimentSerializationContext, self).__init__(name_provider=name_provider)
        self.__label_scaler = label_scaler

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
    def TermsPerContext(self):
        raise NotImplementedError
