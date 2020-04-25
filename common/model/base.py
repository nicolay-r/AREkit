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
