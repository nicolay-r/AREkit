from arekit.common.model.model_io import BaseModelIO


class ExperimentContext(object):

    def __init__(self):
        self.__model_io = None

    @property
    def ModelIO(self):
        """ Provides model paths for the resources utilized during training process.
            The latter is important in Neural Network training process, when there is
            a need to obtain model root directory.
        """
        return self.__model_io

    @property
    def LabelsCount(self):
        raise NotImplementedError()

    def set_model_io(self, model_io):
        """ Providing model_io in experiment data.
        """
        assert(isinstance(model_io, BaseModelIO))
        self.__model_io = model_io
