from arekit.common.model.model_io import BaseModelIO


class NeuralNetworkModelIO(BaseModelIO):
    """
    Provides an API for saving model states
    """

    @property
    def ModelSavePathPrefix(self):
        raise NotImplementedError()
