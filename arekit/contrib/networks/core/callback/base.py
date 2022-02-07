class NetworkCallback(object):

    def on_fit_started(self, operation_cancel):
        pass

    def on_predict_finished(self, pipeline):
        pass

    def on_epoch_finished(self, pipeline, operation_cancel):
        pass