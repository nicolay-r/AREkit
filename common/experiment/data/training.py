from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir


class TrainingData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, labels_scale):
        super(TrainingData, self).__init__(labels_scale)

    @property
    def Evaluator(self):
        raise NotImplementedError()

    def get_experiment_results_dir(self):
        """ Provides directory for model serialized output results.
        """
        raise NotImplementedError()

    def get_model_results_root(self):
        return get_path_of_subfolder_in_experiments_dir(subfolder_name=self.__model_name,
                                                        experiments_dir=self.get_experiment_results_dir())
