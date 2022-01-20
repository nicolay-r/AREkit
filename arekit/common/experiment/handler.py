from arekit.common.experiment.api.ctx_base import DataIO


class ExperimentEngineHandler(object):

    def __init__(self, exp_data):
        assert(isinstance(exp_data, DataIO))
        self._exp_data = exp_data

    def on_before_iteration(self):
        pass

    def on_iteration(self, iter_index):
        pass

    def on_after_iteration(self):
        pass
