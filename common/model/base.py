from arekit.common.experiment.data_type import DataType
from arekit.common.model.model_io import BaseModelIO


class BaseModel(object):
    """
    Base Model
    """

    def __init__(self, io):
        assert(isinstance(io, BaseModelIO))
        self.__io = io

        # TODO. move here evaluator from experiments

    # TODO. move here property to access the evaluator.

    @property
    def IO(self):
        return self.__io

    # TODO. Remove epochs count, since it is related to NeuralNetworks only.
    def run_training(self, epochs_count):
        raise NotImplementedError()

    def predict(self, data_type=DataType.Test):
        raise NotImplementedError()
