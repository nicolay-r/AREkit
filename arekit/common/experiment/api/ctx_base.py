from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.model.model_io import BaseModelIO


class ExperimentContext(object):

    def __init__(self, name_provider, data_folding):
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(name_provider, ExperimentNameProvider))
        self.__model_io = None
        self.__data_folding = data_folding
        self.__name_provider = name_provider

    @property
    def DataFolding(self):
        return self.__data_folding

    @property
    def ModelIO(self):
        """ Provides model paths for the resources utilized during training process.
            The latter is important in Neural Network training process, when there is
            a need to obtain model root directory.
        """
        return self.__model_io

    @property
    def Name(self):
        return self.__name_provider.provide()

    @property
    def LabelsCount(self):
        raise NotImplementedError()

    def set_data_folding(self, data_folding):
        assert(isinstance(data_folding, BaseDataFolding))
        self.__data_folding = data_folding

    def set_model_io(self, model_io):
        """ Providing model_io in experiment data.
        """
        assert(isinstance(model_io, BaseModelIO))
        self.__model_io = model_io
