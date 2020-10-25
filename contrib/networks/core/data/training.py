from arekit.common.experiment.data.training import TrainingData


class NetworkTrainingData(TrainingData):
    """ Specific for NeuralNetworks training data
    """
    
    def __init__(self, labels_scaler):
        super(NetworkTrainingData, self).__init__(labels_scaler)

    @property
    def Callback(self):
        raise NotImplementedError()
