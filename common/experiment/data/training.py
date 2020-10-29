from arekit.common.experiment.data.base import DataIO


class TrainingData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, labels_scaler):
        super(TrainingData, self).__init__(labels_scaler)

    @property
    def Evaluator(self):
        raise NotImplementedError()
