import logging
from collections import OrderedDict

from arekit.contrib.networks.core.callback.utils_hidden_states import save_model_hidden_values
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.network_callback import NetworkCallback
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainingCallback(NetworkCallback):

    def __init__(self, train_acc_limit, log_dir):
        assert(isinstance(train_acc_limit, float) or train_acc_limit is None)
        super(TrainingCallback, self).__init__()
        self.__test_results_exp_history = OrderedDict()
        self.__log_dir = log_dir
        self.__train_acc_limit = train_acc_limit

    # region private methods

    def __is_cancel_needed(self, avg_fit_acc):
        if self.__train_acc_limit is not None and avg_fit_acc >= self.__train_acc_limit:
            logger.info(u"Stop Training Process: avg_fit_acc > {}".format(self.__train_acc_limit))
            return True
        return False

    # endregion

    def on_epoch_finished(self, epoch_index, operation_cancel):
        assert(isinstance(epoch_index, int))
        assert(isinstance(operation_cancel, OperationCancellation))
        super(TrainingCallback, self).on_epoch_finished(epoch_index=epoch_index,
                                                        operation_cancel=operation_cancel)

        item = self._model.from_predicted(MinibatchFittingPipelineItem)

        if item is None:
            return

        assert(isinstance(item, MinibatchFittingPipelineItem))

        if self.__is_cancel_needed(item.TotalFitAccuracy):
            operation_cancel.Cancel()

        if self.__log_dir is None:
            return

        # Saving model hidden values using the related numpy utils.
        save_model_hidden_values(log_dir=self.__log_dir, epoch_index=epoch_index, model=self._model)
