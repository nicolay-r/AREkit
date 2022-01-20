class NetworkCallback(object):

    def on_fit_started(self, operation_cancel):
        # Do nothing by default.
        pass

    def on_epoch_finished(self, pipeline, operation_cancel):
        # Do nothing by default.
        pass