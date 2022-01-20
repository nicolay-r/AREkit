from os.path import join

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.callback.base import NetworkCallback
from arekit.contrib.networks.core.idhv_collection import NetworkInputDependentVariables
from arekit.contrib.networks.core.pipeline.item_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arekit.contrib.networks.core.utils import get_item_from_pipeline


class InputHiddenStatesWriterCallback(NetworkCallback):

    def __init__(self, log_dir, writer):
        super(InputHiddenStatesWriterCallback, self).__init__()
        self.__epochs_passed = 0
        self.__log_dir = log_dir
        self.__writer = writer

    @staticmethod
    def __path_by_var_name(var_name, data_type, epoch_index, log_dir):
        filname = 'idparams_{data}_e{epoch_index}'.format(
            data='{}-{}'.format(var_name, data_type),
            epoch_index=epoch_index)
        return join(log_dir, filname)

    def __save_minibatch_variable_values(self, target, predict_log, var_name):
        assert(isinstance(predict_log, NetworkInputDependentVariables))
        create_dir_if_not_exists(target)
        id_and_value_pairs = list(predict_log.iter_by_parameter_values(var_name))
        id_and_value_pairs = sorted(id_and_value_pairs, key=lambda pair: pair[0])
        self.__writer.write(target=target, data=[pair[1] for pair in id_and_value_pairs])

    def on_epoch_finished(self, pipeline, operation_cancel):
        super(InputHiddenStatesWriterCallback, self).on_epoch_finished(
            pipeline=pipeline,
            operation_cancel=operation_cancel)

        self.__epochs_passed += 1

        pipeline_item = get_item_from_pipeline(pipeline=pipeline,
                                               item_type=MinibatchHiddenFetcherPipelineItem)

        if pipeline_item is None:
            return

        predict_log = pipeline_item.InputDependentParams
        data_type = pipeline_item.DataType

        for var_name in predict_log.iter_var_names():
            target = self.__path_by_var_name(var_name=var_name,
                                             data_type=data_type,
                                             epoch_index=self.__epochs_passed,
                                             log_dir=self.__log_dir)

            self.__save_minibatch_variable_values(
                target=target,
                predict_log=predict_log,
                var_name=var_name)
