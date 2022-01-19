import logging

from arekit.contrib.networks.core.callback_network import NetworkCallback
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.utils import get_item_from_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainingLimiterCallback(NetworkCallback):

    def __init__(self, train_acc_limit):
        assert(isinstance(train_acc_limit, float) or train_acc_limit is None)
        super(TrainingLimiterCallback, self).__init__()
        self.__train_acc_limit = train_acc_limit

    def __is_cancel_needed(self, avg_fit_acc):
        if self.__train_acc_limit is not None and avg_fit_acc >= self.__train_acc_limit:
            logger.info(u"Stop Training Process: avg_fit_acc > {}".format(self.__train_acc_limit))
            return True
        return False

    def on_epoch_finished(self, pipeline, operation_cancel):
        assert(isinstance(operation_cancel, OperationCancellation))
        super(TrainingLimiterCallback, self).on_epoch_finished(pipeline=pipeline,
                                                               operation_cancel=operation_cancel)

        item = get_item_from_pipeline(pipeline=pipeline, item_type=MinibatchFittingPipelineItem)

        if item is None:
            return

        assert(isinstance(item, MinibatchFittingPipelineItem))

        if self.__is_cancel_needed(item.TotalFitAccuracy):
            operation_cancel.Cancel()
