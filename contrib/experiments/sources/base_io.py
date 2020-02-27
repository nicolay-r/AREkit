import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIO
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.networks.network_io import NetworkIO


class BaseIO(NetworkIO):

    def __init__(self, experiments_io, model_name):
        assert(isinstance(experiments_io, BaseExperimentsIO))
        self.__experiments_io = experiments_io
        self.__model_name = model_name

    # region 'get' public methods

    def get_model_filepath(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}'.format(self.__model_name))

    def get_model_root(self):
        return self.__get_model_root()

    def get_word_embedding_filepath(self):
        return self.__experiments_io.get_rusvectores_news_embedding_filepath()

    def get_capitals_filepath(self):
        return self.__experiments_io.get_capitals_filepath()

    def get_states_filepath(self):
        return self.__experiments_io.get_states_filepath()

    # endregion

    # region 'write' methods

    def write_log(self, log_names, log_values):
        assert(isinstance(log_names, list))
        assert(isinstance(log_values, list))
        assert(len(log_names) == len(log_values))

        log_path = os.path.join(self.get_model_root(), u"log.txt")

        with open(log_path, 'w') as f:
            for index, log_value in enumerate(log_values):
                f.write("{}: {}\n".format(log_names[index], log_value))

    # endregion

    @staticmethod
    def read_list_from_lss(filepath):
        """
        Reading lines in lowercase mode
        """
        lines = []
        with open(filepath) as f:
            for line in f.readlines():
                row = line.decode('utf-8')
                row = row.lower().strip()
                lines.append(row)

        return lines

    def create_model_state_filepath(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}.state'.format(self.__model_name))

    # region private methods

    def __get_model_root(self):
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_io=self.__experiments_io)

    def __get_model_states_dir(self):

        result_dir = os.path.join(
            self.__get_model_root(),
            os.path.join(u'model_states/'))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion
