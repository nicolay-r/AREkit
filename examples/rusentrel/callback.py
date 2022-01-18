import logging
from os.path import join

from arekit.contrib.networks.core.callback.np_writer import NpzDataWriter
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.network_callback import NetworkCallback
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.utils import get_item_from_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO. #259. This might be splitted onto
#  - TrainingTerminationCallback. (and move to network/core/callback)
#  - HiddenWriterCallback. (and move to network/core/callback)
class TrainingCallback(NetworkCallback):

    def __init__(self, train_acc_limit, log_dir):
        assert(isinstance(train_acc_limit, float) or train_acc_limit is None)
        super(TrainingCallback, self).__init__()

        # TODO. #259 to Termination.
        self.__train_acc_limit = train_acc_limit

        # TODO. #259 to HiddenWriter.
        self.__log_dir = log_dir
        self.__writer = NpzDataWriter()

    # region private methods

    # TODO. #259 to Termination.
    def __is_cancel_needed(self, avg_fit_acc):
        if self.__train_acc_limit is not None and avg_fit_acc >= self.__train_acc_limit:
            logger.info(u"Stop Training Process: avg_fit_acc > {}".format(self.__train_acc_limit))
            return True
        return False

    # endregion

    # TODO. #259 to HiddenWriter.
    def __target_provider(self, name, epoch_index):
        return join(self.__log_dir, 'hparams_{name}_e{epoch}'.format(name=name, epoch=epoch_index))

    def on_epoch_finished(self, pipeline, operation_cancel):
        assert(isinstance(operation_cancel, OperationCancellation))
        super(TrainingCallback, self).on_epoch_finished(pipeline=pipeline,
                                                        operation_cancel=operation_cancel)

        item = get_item_from_pipeline(pipeline=pipeline, item_type=MinibatchFittingPipelineItem)

        if item is None:
            return

        assert(isinstance(item, MinibatchFittingPipelineItem))

        if self.__is_cancel_needed(item.TotalFitAccuracy):
            operation_cancel.Cancel()

        # TODO. #259 to HiddenWriter.
        if self.__log_dir is None:
            return

        names, values = self._model_ctx.get_hidden_parameters()

        assert(isinstance(names, list))
        assert(isinstance(values, list))
        assert(len(names) == len(values))

        for value_index, name in enumerate(names):
            self.__writer.write(
                data=values[value_index],
                target=self.__target_provider(name=name,
                                              epoch_index=self._training_epochs_passed))
