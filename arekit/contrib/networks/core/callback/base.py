from arekit.contrib.networks.core.model_ctx import TensorflowModelContext


class NetworkCallback(object):

    def __init__(self):
        super(NetworkCallback, self).__init__()
        self._model_ctx = None

    def on_initialized(self, model_ctx):
        assert(isinstance(model_ctx, TensorflowModelContext))
        self._model_ctx = model_ctx

    def on_fit_started(self, operation_cancel):
        # Do nothing by default.
        pass

    def on_epoch_finished(self, pipeline, operation_cancel):
        # Do nothing by default.
        pass