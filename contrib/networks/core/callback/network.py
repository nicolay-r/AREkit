import datetime
import logging
from os.path import join

import numpy as np
import os

from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.callback.model_eval import perform_experiment_evaluation
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.data_handling.predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.output.encoder import NetworkOutputEncoder
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NeuralNetworkCallback(Callback):

    __log_saving_info = True

    __log_train_filename = u"cb_train.log"
    __log_eval_filename = u"cb_eval.log"
    __log_eval_verbose_filename = u"cb_eval_verbose.log"

    __hiddenParams_template = u'hparams_{}_e{}'
    __input_dependent_params_template = u'idparams_{}_e{}'

    def __init__(self):

        self.__model = None
        self.__log_dir = None
        self.__experiment = None

        self._test_on_epochs = None
        self.__cv_index = None

        self.__train_log_file = None
        self.__eval_log_file = None
        self.__eval_verbose_log_file = None

        self.reset_experiment_dependent_parameters()

        self.__key_save_hidden_parameters = True
        self.__key_stop_training_by_cost = False

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

        message = "{}: Epoch: {}: avg_fit_cost: {:.3f}, avg_fit_acc: {:.3f}".format(
            str(datetime.datetime.now()),
            epoch_index,
            avg_fit_cost,
            avg_fit_acc)

        # Providing information into main logger.
        logger.info(message)

        # Duplicate the related information in separate log file.
        self.__train_log_file.write("{}\n".format(message))

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
            variable_path = os.path.join(self.__log_dir, self.__hiddenParams_template.format(name, epoch_index))
            if self.__log_saving_info:
                logger.info(u"Save hidden values: {}".format(variable_path))
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

        vars_path = join(self.__log_dir,
                         self.__input_dependent_params_template.format(
                             u'{}-{}'.format(var_name, data_type),
                             epoch_index))
        create_dir_if_not_exists(vars_path)

        if self.__log_saving_info:
            msg = u"Save input dependent hidden values in a list using np.savez: {}".format(vars_path)
            logger.info(msg)

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
        result_filepath = self.__experiment.ExperimentIO.get_output_model_results_filepath(data_type=data_type,
                                                                                           epoch_index=epoch_index)

        # Save output
        output.to_tsv(filepath=result_filepath)

        # Convert output to result.
        result = perform_experiment_evaluation(experiment=self.__experiment,
                                               data_type=data_type,
                                               epoch_index=epoch_index,
                                               labels_formatter=RuSentRelLabelsFormatter())

        eval_verbose_msg = self.__create_verbose_eval_results_msg(eval_result=result,
                                                                  data_type=data_type,
                                                                  epoch_index=epoch_index)

        eval_msg = self.__create_overall_eval_results_msg(eval_result=result,
                                                          data_type=data_type,
                                                          epoch_index=epoch_index)

        # Writing evaluation logging results.
        logger.info(eval_verbose_msg)
        logger.info(eval_msg)

        # Separate logging information by files.
        self.__eval_log_file.write(eval_msg)
        self.__eval_verbose_log_file.write(eval_verbose_msg)

        self._save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            epoch_index=epoch_index)

        return result

    # endregion

    # region logging

    @staticmethod
    def __create_verbose_eval_results_msg(eval_result, data_type, epoch_index):
        assert(isinstance(eval_result, TwoClassEvalResult))
        title = u"Stat for [{dtype}], e={epoch}:".format(dtype=data_type, epoch=epoch_index)
        contents = [u"{doc_id}: {result}".format(doc_id=doc_id, result=result)
                    for doc_id, result in eval_result.iter_document_results()]
        return u'\n'.join([title, contents])

    @staticmethod
    def __create_overall_eval_results_msg(eval_result, data_type, epoch_index):
        assert(isinstance(eval_result, TwoClassEvalResult))
        title = u"Stat for '[{dtype}]', e={epoch}".format(dtype=data_type, epoch=epoch_index)
        params = [u"{}: {}".format(metric_name, round(value, 2))
                  for metric_name, value in eval_result.iter_results()]
        contents = u"; ".join(params)
        return u'\n'.join([title, contents])

    # endregion

    def __enter__(self):
        assert(self.__log_dir is not None)

        train_log_filepath = join(self.__log_dir, self.__log_train_filename)
        eval_log_filepath = join(self.__log_dir, self.__log_eval_filename)
        eval_verbose_log_filepath = join(self.__log_dir, self.__log_eval_verbose_filename)

        create_dir_if_not_exists(train_log_filepath)
        create_dir_if_not_exists(eval_log_filepath)
        create_dir_if_not_exists(eval_verbose_log_filepath)

        self.__train_log_file = open(train_log_filepath, u"w")
        self.__eval_log_file = open(eval_log_filepath, u"w")
        self.__eval_verbose_log_file = open(eval_verbose_log_filepath, u"w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__train_log_file is not None:
            self.__train_log_file.close()

        if self.__eval_log_file is not None:
            self.__eval_log_file.close()

        if self.__eval_verbose_log_file is not None:
            self.__eval_verbose_log_file.close()
