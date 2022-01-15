import logging
from collections import OrderedDict

from arekit.contrib.networks.core.callback.utils_hidden_states import save_model_hidden_values
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.network_callback import NetworkCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainingCallback(NetworkCallback):

    def __init__(self, train_acc_limit, epochs_count, log_dir):
        assert(isinstance(train_acc_limit, float) or train_acc_limit is None)
        super(TrainingCallback, self).__init__(epochs_count)
        self.__model = None
        self.__test_results_exp_history = OrderedDict()
        self.__log_dir = log_dir
        self.__train_acc_limit = train_acc_limit

    # region public methods

    def on_initialized(self, model):
        assert(isinstance(model, BaseTensorflowModel))
        self.__model = model

    # endregion

    # region private methods

    def __is_cancel_needed(self, avg_fit_acc):
        if self.__train_acc_limit is not None and avg_fit_acc >= self.__train_acc_limit:
            logger.info(u"Stop feeding process: avg_fit_acc > {}".format(self.__train_acc_limit))
            return True
        return False

    # endregion

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(epoch_index, int))
        assert(isinstance(operation_cancel, OperationCancellation))

        super(TrainingCallback, self).on_epoch_finished(avg_fit_cost=avg_fit_cost,
                                                        avg_fit_acc=avg_fit_acc,
                                                        epoch_index=epoch_index,
                                                        operation_cancel=operation_cancel)

        if self.__is_cancel_needed(avg_fit_acc):
            operation_cancel.Cancel()

        if self.__log_dir is None:
            return

        # Saving model hidden values using the related numpy utils.
        save_model_hidden_values(log_dir=self.__log_dir, epoch_index=epoch_index, model=self.__model)
