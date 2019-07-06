class Callback(object):

    def on_initialized(self, network):
        pass

    def on_epoch_finished(self, avg_cost, avg_acc, epoch_index, operation_cancel):
        pass

    def on_fit_finished(self):
        pass
