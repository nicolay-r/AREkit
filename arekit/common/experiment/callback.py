import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Callback(object):

    def __init__(self):
        self._experiment = None

    def set_experiment(self, experiment):
        self._experiment = experiment

    def on_initialized(self, network):
        # Do nothing by default.
        pass

    def on_experiment_iteration_begin(self):
        # Do nothing by default.
        pass

    def on_fit_started(self, operation_cancel):
        # Do nothing by default.
        pass

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        # Do nothing by default.
        pass

    def on_fit_finished(self):
        # Do nothing by default.
        pass

    def on_experiment_finished(self):
        # Do nothing by default.
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
