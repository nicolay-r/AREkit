import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExperimentCallback(object):

    def on_experiment_iteration_begin(self):
        # Do nothing by default.
        pass

    def on_experiment_finished(self):
        # Do nothing by default.
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
