import datetime
import logging

import numpy as np
import os

from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.utils import create_dir_if_not_exists
from arekit.networks.callback.base import Callback
from arekit.networks.callback.model_eval import perform_experiment_evaluation
from arekit.networks.cancellation import OperationCancellation
from arekit.networks.io_utils import NetworkIOUtils
from arekit.networks.model import BaseTensorflowModel
from arekit.networks.output.encoder import NetworkOutputEncoder
from arekit.networks.data_handling.predict_log import NetworkInputDependentVariables
from arekit.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NeuralNetworkCallback(Callback):

    VocabularyOutputFilePathInLogDir = u'vocab.txt'
    HiddenParamsTemplate = u'hparams_{}_e{}'
    InputDependentParamsTemplate = u'idparams_{}_e{}'
    PredictVerbosePerFileStatistic = True
    FitEpochCompleted = True

    def __init__(self):

        self.__model = None
        self.__log_dir = None
        self.__experiment = None

        self._test_on_epochs = None
        self.__cv_index = None

        self.reset_experiment_dependent_parameters()

        self.__key_save_hidden_parameters = True
        self.__key_stop_training_by_cost = False
        self.__debug_save_info = False

        self.__train_doc_ids = None

    @property
    def Epochs(self):
        return max(self._test_on_epochs)

    # region event handlers

    def set_log_dir(self, log_dir):
        assert(isinstance(log_dir, unicode))
        self.__log_dir = log_dir

    def set_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self.__experiment = experiment

    def set_cv_index(self, cv_index):
        assert (isinstance(cv_index, int))
        self.__cv_index = cv_index

    def on_initialized(self, model):
        assert(isinstance(model, BaseTensorflowModel))
        self.__model = model

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(operation_cancel, OperationCancellation))

        if self.FitEpochCompleted:
            logger.info("{}: Epoch: {}: avg_fit_cost: {:.3f}, avg_fit_acc: {:.3f}".format(
                str(datetime.datetime.now()),
                epoch_index,
                avg_fit_cost,
                avg_fit_acc))

        if (epoch_index not in self._test_on_epochs) and (not operation_cancel.IsCancelled):
            return

        self.__save_model_hidden_values(epoch_index)

    # endregion

    # region 'set' methods

    def set_key_save_hidden_parameters(self, value):
        assert(isinstance(value, bool))
        self.__key_save_hidden_parameters = value

    def set_test_on_epochs(self, value):
        assert(isinstance(value, list))
        self._test_on_epochs = value

    # endregion

    # region private methods

    def __save_model_hidden_values(self, epoch_index):

        if not self.__key_save_hidden_parameters:
            return

        names, values = self.__model.get_hidden_parameters()

        assert(isinstance(names, list))
        assert(isinstance(values, list))
        assert(len(names) == len(values))

        for i, name in enumerate(names):
            variable_path = os.path.join(self.__log_dir, self.HiddenParamsTemplate.format(name, epoch_index))
            if self.__debug_save_info:
                print "Save hidden values: {}".format(variable_path)
            create_dir_if_not_exists(variable_path)
            np.save(variable_path, values[i])

    # endregion

    # region extra functionality

    def _save_minibatch_all_input_dependent_hidden_values(self, data_type, epoch_index, predict_log):
        assert(isinstance(predict_log, NetworkInputDependentVariables))

        if not self.__key_save_hidden_parameters:
            return

        for var_name in predict_log.iter_var_names():
            self._save_minibatch_variable_values(data_type=data_type,
                                                 epoch_index=epoch_index,
                                                 predict_log=predict_log,
                                                 var_name=var_name)

    def _save_minibatch_variable_values(self, data_type, epoch_index, predict_log, var_name):
        assert(isinstance(predict_log, NetworkInputDependentVariables))
        assert(isinstance(var_name, unicode))

        if not self.__key_save_hidden_parameters:
            return

        vars_path = os.path.join(self.__log_dir,
                                 self.InputDependentParamsTemplate.format(
                                     '{}-{}'.format(var_name, data_type),
                                     epoch_index))
        create_dir_if_not_exists(vars_path)

        if self.__debug_save_info:
            print "Save input dependent hidden values in a list using np.savez: {}".format(vars_path)

        id_and_value_pairs = list(predict_log.iter_by_parameter_values(var_name))
        id_and_value_pairs = sorted(id_and_value_pairs, key=lambda pair: pair[0])
        np.savez(vars_path, [pair[1] for pair in id_and_value_pairs])

    # endregion

    # region evaluation

    def _evaluate_model(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        idhp, output = self.__model.predict(data_type=data_type)

        assert(isinstance(idhp, NetworkInputDependentVariables))
        assert(isinstance(output, NetworkOutputEncoder))

        # Crate filepath
        result_filepath = NetworkIOUtils.get_output_results_filepath(experiment=self.__experiment,
                                                                     data_type=data_type,
                                                                     epoch_index=epoch_index)

        # Save output
        output.to_tsv(filepath=result_filepath)

        # Convert output to result.
        result = perform_experiment_evaluation(experiment=self.__experiment,
                                               data_type=data_type,
                                               epoch_index=epoch_index,
                                               labels_formatter=RuSentRelLabelsFormatter())

        if self.PredictVerbosePerFileStatistic:
            self._print_verbose_eval_results(eval_result=result,
                                             data_type=data_type,
                                             epoch_index=epoch_index)

        self._print_overall_results(eval_result=result,
                                    data_type=data_type,
                                    epoch_index=epoch_index)

        self._save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            epoch_index=epoch_index)

        return result

    # endregion

    # region logging

    @staticmethod
    def _print_verbose_eval_results(eval_result, data_type, epoch_index):
        assert(isinstance(eval_result, TwoClassEvalResult))
        logger.info("Stat for {dtype}, e={epoch}:".format(dtype=data_type, epoch=epoch_index))
        for doc_id, result in eval_result.iter_document_results():
            logger.info(doc_id, result)

    @staticmethod
    def _print_overall_results(eval_result, data_type, epoch_index):
        assert(isinstance(eval_result, TwoClassEvalResult))

        logger.info("Stat for '{dtype}', e={epoch}".format(dtype=data_type, epoch=epoch_index))
        params = ["{}: {}".format(metric_name, round(value, 2))
                  for metric_name, value in eval_result.iter_results()]
        logger.info("; ".join(params))

    # endregion
