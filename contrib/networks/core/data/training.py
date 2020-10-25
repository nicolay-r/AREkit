from arekit.common.experiment.data.training import TrainingData


class NetworkTrainingData(TrainingData):
    """ Specific for NeuralNetworks training data
    """
    
    def __init__(self, labels_scale):
        super(NetworkTrainingData, self).__init__(labels_scale)

    @property
    def Callback(self):
        raise NotImplementedError()
