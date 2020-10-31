from arekit.common.experiment.engine.utils import setup_logger
from arekit.common.experiment.formats.base import BaseExperiment


class ExperimentEngine(object):
    """ Represents a CV-Based Experiment engine
        The core method is `run`, which assumes to perform iteration per every
        iteration in and runs handler during the latter.
    """

    def __init__(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self._experiment = experiment
        self._logger = setup_logger()

    # region protected methods

    def _handle_iteration(self, iter_index):
        raise NotImplementedError()

    def _before_running(self):
        """ Optional method that allows to implement actions before engine started.
        """
        pass

    # endregion

    def run(self):
        """ Running cv_index iteration and calling handler during every iteration.
        """

        self._before_running()

        for iter_index, _ in enumerate(self._experiment.DocumentOperations.DataFolding.iter_states()):
            self._handle_iteration(iter_index)
