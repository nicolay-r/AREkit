from arekit.common.data_type import DataType
from arekit.common.model.model_io import BaseModelIO


class BaseModel(object):
    """
    Base Model
    """

    def __init__(self, io):
        assert(isinstance(io, BaseModelIO))
        self.__io = io

    @property
    def IO(self):
        return self.__io

    # TODO. Remove epochs count, since it is related to NeuralNetworks only.
    def run_training(self, epochs_count, load_model=False):
        raise NotImplementedError()

    def predict(self, dest_data_type=DataType.Test, doc_ids_set=None):
        raise NotImplementedError()
