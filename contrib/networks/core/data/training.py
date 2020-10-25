from arekit.common.experiment.data.training import TrainingData
from arekit.common.experiment.engine.utils import rm_dir_contents
from arekit.common.model.model_io import BaseModelIO


class NetworkTrainingData(TrainingData):
    """ Specific for NeuralNetworks training data
    """
    
    def __init__(self, labels_scale):
        super(NetworkTrainingData, self).__init__(labels_scale)

    @property
    def Callback(self):
        raise NotImplementedError()

    def prepare_model_root(self, logger, rm_contents=True):

        if not rm_contents:
            return

        model_io = self.ModelIO
        assert(isinstance(model_io, BaseModelIO))
        rm_dir_contents(model_io.ModelRoot,
                        logger=logger)
