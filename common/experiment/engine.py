import logging
from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment


class CVBasedExperimentEngine(object):
    """ Represents a CV-Based Experiment engine
        The core method is `run`, which assumes to perform iteration per every
        cross-validation iteration in order to run `handle_cv_index` handler.
    """

    def __init__(self, experiment):
        assert(isinstance(experiment, CVBasedExperiment))
        self._experiment = experiment
        self._logger = self.__setup_logger()

    # region private methods

    @staticmethod
    def __setup_logger():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    # endregion

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