class Callback(object):

    def on_initialized(self, network):
        pass

    @property
    def Epochs(self):
        raise NotImplementedError()

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        pass

    def on_fit_finished(self):
        pass

    def set_test_on_epochs(self, value):
        raise NotImplementedError()

    def set_cv_index(self, cv_index):
        raise NotImplementedError()

    def reset_experiment_dependent_parameters(self):
        raise NotImplementedError()