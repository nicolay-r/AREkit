from arekit.common.experiment.api.ctx_base import DataIO
from arekit.common.experiment.callback import Callback


class TrainingData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, labels_count):
        super(TrainingData, self).__init__()
        self.__labels_count = labels_count
        self.__callback = Callback()

    @property
    def LabelsCount(self):
        return self.__labels_count

    @property
    def Evaluator(self):
        raise NotImplementedError()

    @property
    def Callback(self):
        return self.__callback
