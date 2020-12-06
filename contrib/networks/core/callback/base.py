class Callback(object):

    @property
    def Epochs(self):
        raise NotImplementedError()

    def set_experiment(self, experiment):
        raise NotImplementedError()

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
