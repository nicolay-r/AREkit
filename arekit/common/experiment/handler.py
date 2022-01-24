from arekit.common.experiment.api.ctx_base import ExperimentContext


class ExperimentIterationHandler(object):

    def __init__(self, exp_ctx):
        assert(isinstance(exp_ctx, ExperimentContext))
        self._exp_ctx = exp_ctx

    def on_before_iteration(self):
        pass

    def on_iteration(self, iter_index):
        pass

    def on_after_iteration(self):
        pass
