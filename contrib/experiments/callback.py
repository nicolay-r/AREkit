import datetime

import numpy as np
import os

from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.utils import create_dir_if_not_exists
from arekit.networks.callback import Callback
from arekit.networks.cancellation import OperationCancellation
from arekit.networks.context.debug import DebugKeys
from arekit.networks.tf_model import TensorflowModel
from arekit.networks.data_type import DataType
from arekit.networks.predict_log import NetworkInputDependentVariables


class CustomCallback(Callback):

    VocabularyOutputFilePathInLogDir = u'vocab.txt'
    HiddenParamsTemplate = u'hparams_{}_e{}'
    InputDependentParamsTemplate = u'idparams_{}_e{}'
    PredictVerbosePerFileStatistic = True

    def __init__(self, log_dir):

        self.__model = None
        self.__test_on_epochs = None
        self.__log_dir = log_dir

        self.__costs_history = None
        self.__reset_experiment_dependent_parameters()

        self.__costs_window = 5

        self.__key_save_hidden_parameters = True
        self.__key_stop_training_by_cost = False
        self.__debug_save_info = False

        self.__cancellation_acc_bound = 0.99
        self.__cancellation_f1_train_bound = 0.85

        self.__train_doc_ids = None

    @property
    def Epochs(self):
        return max(self.__test_on_epochs)

    def reset_experiment_dependent_parameters(self):
        self.__reset_experiment_dependent_parameters()

    # region event handlers

    def on_initialized(self, model):
        assert(isinstance(model, TensorflowModel))
        self.__model = model

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(operation_cancel, OperationCancellation))

        if DebugKeys.FitEpochCompleted:
            print "{}: Epoch: {}: avg_fit_cost: {:.3f}, avg_fit_acc: {:.3f}".format(
                str(datetime.datetime.now()),
                epoch_index,
                avg_fit_cost,
                avg_fit_acc)

        if avg_fit_acc >= self.__cancellation_acc_bound:
            print "Stop training process: avg_fit_acc > {}".format(self.__cancellation_acc_bound)
            operation_cancel.Cancel()

        if (epoch_index not in self.__test_on_epochs) and (not operation_cancel.IsCancelled):
            return

        result_train = self.__process_for_data_type(data_type=DataType.Train,
                                                    epoch_index=epoch_index)

        f1_train = result_train.get_result_by_metric(result_train.C_F1)
        if f1_train >= self.__cancellation_f1_train_bound:
            print "Stop training process: F1-train ({}) > {}".format(
                round(f1_train, 3),
                self.__cancellation_f1_train_bound)
            operation_cancel.Cancel()

        self.__process_for_data_type(data_type=DataType.Test,
                                     epoch_index=epoch_index)

        if self.__key_stop_training_by_cost:
            if not self.__check_costs_still_improving(avg_fit_cost):
                print "Cancelling: cost becomes greater than min value {} epochs ago.".format(
                    self.__costs_window)
                operation_cancel.Cancel()

        self.__save_model_hidden_values(epoch_index)
        self.__save_model_vocabulary()
        self.__costs_history.append(avg_fit_cost)

    # endregion

    # region 'set' methods

    def set_key_stop_training_by_cost(self, value):
        assert(isinstance(value, bool))
        self.__key_stop_training_by_cost = value

    def set_key_save_hidden_parameters(self, value):
        assert(isinstance(value, bool))
        self.__key_save_hidden_parameters = value

    def set_test_on_epochs(self, value):
        assert(isinstance(value, list))
        self.__test_on_epochs = value

    def set_cancellation_acc_bound(self, value):
        assert(isinstance(value, float))
        self.__cancellation_acc_bound = value

    def set_cancellation_f1_train_bound(self, value):
        assert(isinstance(value, float))
        self.__cancellation_f1_train_bound = value

    def set_test_doc_ids_list(self, value):
        """
        Due to the train set could be large and F1 evaluation might take a lot of time.
        This parameter allows to limit amount of documents for F1-train evaluation.

        value: list of None
            list of doc_ids or None
        """
        assert(isinstance(value, list) or value is None)
        self.__train_doc_ids = value

    # endregion

    # region private methods

    def __reset_experiment_dependent_parameters(self):
        self.__costs_history = []

    def __check_costs_still_improving(self, avg_cost):

        history_len = len(self.__costs_history)

        if history_len <= self.__costs_window:
            return True

        return avg_cost < min(self.__costs_history[:history_len - self.__costs_window])

    def __process_for_data_type(self, data_type, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(epoch_index, int))

        result, idhp = self.__model.predict(
            dest_data_type=data_type,
            doc_ids_set=None if data_type == DataType.Test else self.__train_doc_ids)

        assert(isinstance(idhp, NetworkInputDependentVariables))
        assert(isinstance(result, TwoClassEvalResult))

        if self.PredictVerbosePerFileStatistic:
            self.__print_verbose_eval_results(result, data_type)

        self.__print_overall_results(result, data_type)
        self.__save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            epoch_index=epoch_index)

        return result

    def __save_model_vocabulary(self):
        assert(isinstance(self.__model, TensorflowModel))

        if not self.__key_save_hidden_parameters:
            return

        vocab_path = os.path.join(self.__log_dir, self.VocabularyOutputFilePathInLogDir)
        np.savez(vocab_path, list(self.__model.iter_inner_input_vocabulary()))

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

    def __save_minibatch_all_input_dependent_hidden_values(self, data_type, epoch_index, predict_log):
        assert(isinstance(predict_log, NetworkInputDependentVariables))

        if not self.__key_save_hidden_parameters:
            return

        for var_name in predict_log.iter_var_names():
            self.__save_minibatch_variable_values(data_type=data_type,
                                                  epoch_index=epoch_index,
                                                  predict_log=predict_log,
                                                  var_name=var_name)

    def __save_minibatch_variable_values(self, data_type, epoch_index, predict_log, var_name):
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

    @staticmethod
    def __print_verbose_eval_results(eval_result, data_type):
        assert(isinstance(eval_result, TwoClassEvalResult))

        print "Verbose statistic for {}:".format(data_type)
        for doc_id, result in eval_result.iter_document_results():
            print doc_id, result

    @staticmethod
    def __print_overall_results(eval_result, data_type):
        assert(isinstance(eval_result, TwoClassEvalResult))

        print "Overall statistic for '{}' type:".format(data_type)

        params = ["{}: {}".format(metric_name, round(value, 2))
                  for metric_name, value in eval_result.iter_results()]

        print "; ".join(params)

    # endregion
