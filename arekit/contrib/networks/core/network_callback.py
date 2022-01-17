import collections
import logging
from datetime import datetime

from arekit.common.experiment.callback import Callback
from arekit.common.utils import progress_bar_defined
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NetworkCallback(Callback):
    """ Represent network callback which provides
        wrappers for batches iterations
        and epoch termination notifications.
    """

    def __init__(self):
        super(NetworkCallback, self).__init__()
        self.__training_epochs_passed = 0
        self._model = None

    def on_initialized(self, model):
        super(NetworkCallback, self).on_initialized(model)
        self._model = model

    @staticmethod
    def __create_epoch_stat(epoch_index, avg_fit_cost, avg_fit_acc):
        """ Providing epoch training results notification.
        """
        kv_fmt = u"{k}: {v}"
        time = str(datetime.now())
        epochs = kv_fmt.format(k="Epoch", v=format(epoch_index))
        avg_fc = kv_fmt.format(k="avg_fit_cost", v=round(avg_fit_cost, 3))
        avg_ac = kv_fmt.format(k="avg_fig_acc", v=round(avg_fit_acc, 3))
        return u"{time}: {epochs}: {avg_fc}, {avg_ac}".format(
            time=time, epochs=epochs, avg_fc=avg_fc, avg_ac=avg_ac)

    def on_epoch_finished(self, epoch_index, operation_cancel):
        assert(isinstance(epoch_index, int))
        assert(isinstance(operation_cancel, OperationCancellation))

        item = self._model.from_fitted(MinibatchFittingPipelineItem)
        self.__training_epochs_passed += 1

        if item is None:
            return

        assert(isinstance(item, MinibatchFittingPipelineItem))

        super(NetworkCallback, self).on_epoch_finished(epoch_index=epoch_index,
                                                       operation_cancel=operation_cancel)

        message = self.__create_epoch_stat(epoch_index=epoch_index,
                                           avg_fit_cost=item.TotalFitCost,
                                           avg_fit_acc=item.TotalFitAccuracy)

        # Providing information into main logger.
        logger.info(message)

    def handle_batches_iter(self, batches_iter, total, prefix, unit='mbs'):
        """ Do wrapping progress notification.
        """
        assert(isinstance(batches_iter, collections.Iterable))
        assert(isinstance(unit, str))
        assert(isinstance(prefix, str))
        desc = "{prefix} e={epoch}".format(prefix=prefix, epoch=self.__training_epochs_passed)
        return progress_bar_defined(iterable=batches_iter, unit=unit, total=total, desc=desc)
