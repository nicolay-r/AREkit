import logging
from datetime import datetime

from arekit.contrib.networks.core.callback.base import NetworkCallback
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.utils import get_item_from_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainingStatProviderCallback(NetworkCallback):
    """ Represent network callback which provides
        wrappers for batches iterations
        and epoch termination notifications.
    """

    def __init__(self):
        super(TrainingStatProviderCallback, self).__init__()
        self._training_epochs_passed = 0
        self._model_ctx = None

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

    def on_epoch_finished(self, pipeline, operation_cancel):
        assert(isinstance(operation_cancel, OperationCancellation))

        item = get_item_from_pipeline(pipeline=pipeline, item_type=MinibatchFittingPipelineItem)

        self._training_epochs_passed += 1

        if item is None:
            return

        assert(isinstance(item, MinibatchFittingPipelineItem))

        message = self.__create_epoch_stat(epoch_index=self._training_epochs_passed,
                                           avg_fit_cost=item.TotalFitCost,
                                           avg_fit_acc=item.TotalFitAccuracy)

        # Providing information into main logger.
        logger.info(message)
