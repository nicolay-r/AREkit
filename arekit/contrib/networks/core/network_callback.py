import collections
import logging
from datetime import datetime

from arekit.common.experiment.callback import Callback
from arekit.common.utils import progress_bar_defined

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

    @staticmethod
    def __create_epoch_stat(epoch_index, avg_fit_cost, avg_fit_acc):
        """ Providing epoch training results notification.
        """
        kv_fmt = u"{k}: {v}"
        time = str(datetime.now())
        epochs = kv_fmt.format(k="Epoch", v=format(epoch_index))
        avg_fc = kv_fmt.format(k="avg_fit_cost", v=avg_fit_cost)
        avg_ac = kv_fmt.format(k="avg_fig_acc", v=avg_fit_acc)
        return u"{time}: {epochs}: {avg_fc}, {avg_ac}".format(
            time=time, epochs=epochs, avg_fc=avg_fc, avg_ac=avg_ac)

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        super(NetworkCallback, self).on_epoch_finished(avg_fit_acc=avg_fit_acc,
                                                       avg_fit_cost=avg_fit_cost,
                                                       epoch_index=epoch_index,
                                                       operation_cancel=operation_cancel)

        message = self.__create_epoch_stat(epoch_index=epoch_index,
                                           avg_fit_cost=avg_fit_cost,
                                           avg_fit_acc=avg_fit_acc)

        # Providing information into main logger.
        logger.info(message)
        self.__training_epochs_passed += 1

    def handle_batches_iter(self, batches_iter, total, prefix, unit='mbs'):
        """ Do wrapping progress notification.
        """
        assert(isinstance(batches_iter, collections.Iterable))
        assert(isinstance(unit, str))
        assert(isinstance(prefix, str))
        desc = "{prefix} e={epoch}".format(prefix=prefix, epoch=self.__training_epochs_passed)
        return progress_bar_defined(iterable=batches_iter, unit=unit, total=total, desc=desc)
