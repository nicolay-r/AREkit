import datetime

import numpy as np
import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.networks.callback import Callback
from arekit.networks.cancellation import OperationCancellation
from arekit.networks.tf_models.model import TensorflowModel


class ExperimentCallback(Callback):

    VocabularyOutputFilePathInLogDir = u'vocab.txt'
    HiddenParamsTemplate = u'hparams_{}_e{}'
    InputDependentParamsTemplate = u'idparams_{}_e{}'
    PredictVerbosePerFileStatistic = True
    FitEpochCompleted = True

    def __init__(self):

        self.__model = None
        self.__test_on_epochs = None
        self.__log_dir = None

        self.reset_experiment_dependent_parameters()

        self.__key_save_hidden_parameters = True
        self.__key_stop_training_by_cost = False
        self.__debug_save_info = False

        self.__train_doc_ids = None

    @property
    def Epochs(self):
        return max(self.__test_on_epochs)

    # region event handlers

    def set_log_dir(self, log_dir):
        assert(isinstance(log_dir, unicode))
        self.__log_dir = log_dir

    def on_initialized(self, model):
        assert(isinstance(model, TensorflowModel))
        self.__model = model

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(operation_cancel, OperationCancellation))

        if self.FitEpochCompleted:
            print "{}: Epoch: {}: avg_fit_cost: {:.3f}, avg_fit_acc: {:.3f}".format(
                str(datetime.datetime.now()),
                epoch_index,
                avg_fit_cost,
                avg_fit_acc)

        if (epoch_index not in self.__test_on_epochs) and (not operation_cancel.IsCancelled):
            return

        self.__save_model_hidden_values(epoch_index)
        # TODO. Save earlier, before model actually start.
        self.__save_model_vocabulary()

    # endregion

    # region 'set' methods

    def set_key_save_hidden_parameters(self, value):
        assert(isinstance(value, bool))
        self.__key_save_hidden_parameters = value

    def set_test_on_epochs(self, value):
        assert(isinstance(value, list))
        self.__test_on_epochs = value

    # endregion

    # region private methods

    def __save_model_vocabulary(self):
        assert(isinstance(self.__model, TensorflowModel))

        if not self.__key_save_hidden_parameters:
            return

        vocab_path = os.path.join(self.__log_dir, self.VocabularyOutputFilePathInLogDir)
        # TODO. This should be saved earlier.
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

    # endregion
