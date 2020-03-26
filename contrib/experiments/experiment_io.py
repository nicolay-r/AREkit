import os
from os import path

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.operations.opinions import OpinionOperations
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir, rm_dir_contents
from arekit.networks.nn_io import NeuralNetworkIO


class BaseExperimentNeuralNetworkIO(NeuralNetworkIO, OpinionOperations):

    def __init__(self, data_io, model_name):
        assert(isinstance(data_io, DataIO))

        OpinionOperations.__init__(self)

        self.__data_io = data_io
        self.__model_name = model_name

    # region Properties

    @property
    def DataIO(self):
        return self.__data_io

    # endregion

    # region implemented nn_io

    @property
    def ModelSavePathPrefix(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}'.format(self.__model_name))

    def __get_model_states_dir(self):
        result_dir = os.path.join(
            self.get_model_root(),
            os.path.join(u'model_states/'))

        create_dir_if_not_exists(result_dir)
        return result_dir

    def get_model_root(self):
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.__data_io.get_experiments_dir())

    # endregion

    def prepare_model_root(self, rm_contents=True):

        if not rm_contents:
            return

        rm_dir_contents(self.get_model_root())

    def get_logfile_dir(self):
        return path.join(self.get_model_root(), u"log/")

    # region Document Operations

    def read_parsed_news(self, doc_id):
        raise NotImplementedError()

    def iter_data_indices(self, data_type):
        raise NotImplementedError()

    # endregion
