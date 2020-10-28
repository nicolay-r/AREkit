from arekit.common.experiment.engine.utils import setup_logger
from arekit.common.experiment.formats.base import BaseExperiment


class CVBasedExperimentEngine(object):
    """ Represents a CV-Based Experiment engine
        The core method is `run`, which assumes to perform iteration per every
        cross-validation iteration in order to run `handle_cv_index` handler.
    """

    def __init__(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self._experiment = experiment
        self._logger = setup_logger()

    # region protected methods

    def _handle_cv_index(self, cv_index):
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

        for cv_index in range(self._experiment.DataIO.CVFoldingAlgorithm.CVCount):
            self._experiment.DataIO.CVFoldingAlgorithm.set_iteration_index(cv_index)
            self._handle_cv_index(cv_index)