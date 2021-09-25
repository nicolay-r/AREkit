import logging

from arekit.common.experiment.api.base import BaseExperiment


class ExperimentEngine(object):
    """ Represents a CV-Based Experiment engine
        The core method is `run`, which assumes to perform iteration per every
        iteration in and runs handler during the latter.
    """

    def __init__(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self._experiment = experiment
        self._logger = self.__create_logger()

    @staticmethod
    def __create_logger():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    # region protected methods

    def _handle_iteration(self, iter_index):
        raise NotImplementedError()

    def _before_running(self):
        """ Optional method that allows to implement actions before experiment iterations.
        """
        pass

    def _after_running(self):
        """ Optional method that allows to implement actions after experiment iterations.
        """

    # endregion

    def run(self):
        """ Running cv_index iteration and calling handler during every iteration.
        """

        self._before_running()

        for iter_index, _ in enumerate(self._experiment.DocumentOperations.DataFolding.iter_states()):
            self._handle_iteration(iter_index)

        self._after_running()
