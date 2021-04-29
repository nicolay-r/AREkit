from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.model.model_io import BaseModelIO


class DataIO(object):
    """ This base class aggregates all the data necessary for
        cv-based experiment organization
        (data-serialization, training, etc.).
    """

    def __init__(self, labels_scaler, stemmer):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        self.__labels_scale = labels_scaler
        self.__stemmer = stemmer
        self.__model_io = None

    @property
    def LabelsScaler(self):
        """ Declares the amount of labels utilized in experiment. The latter
            is necessary for conversions from int (uint) to Labels and vice versa.
        """
        return self.__labels_scale

    @property
    def ModelIO(self):
        """ Provides model paths for the resources utilized during training process.
            The latter is important in Neural Network training process, when there is
            a need to obtain model root directory.
        """
        return self.__model_io

    @property
    def Stemmer(self):
        return self.__stemmer

    # region not implemented properties

    @property
    def OpinionFormatter(self):
        """ Corresponds to `OpinionCollectionsFormatter` instance
        """
        raise NotImplementedError()

    # endregion

    def set_model_io(self, model_io):
        """ Providing model_io in experiment data.
        """
        assert(isinstance(model_io, BaseModelIO))
        self.__model_io = model_io
