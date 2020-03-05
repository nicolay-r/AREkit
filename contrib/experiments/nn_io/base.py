import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIOUtils
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.networks.nn_io import NeuralNetworkIO
from arekit.processing.lemmatization.base import Stemmer


class BaseExperimentNeuralNetworkIO(NeuralNetworkIO):

    def __init__(self, experiments_io, model_name):
        assert(isinstance(experiments_io, BaseExperimentsIOUtils))
        self.__experiments_io = experiments_io
        self.__model_name = model_name
        self.__synonyms = None

    @property
    def SynonymsCollection(self):
        return self.__synonyms

    # region 'get' public methods

    def get_model_filepath(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}'.format(self.__model_name))

    def get_model_root(self):
        return self.__get_model_root()

    @property
    def ExperimentsIO(self):
        return self.__experiments_io

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

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):
        raise NotImplementedError()

    def read_neutral_opinion_collection(self, doc_id, data_type):
        raise NotImplementedError()

    def read_synonyms_collection(self, stemmer):
        raise NotImplementedError()

    def init_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__synonyms = self.read_synonyms_collection(stemmer=stemmer)

    def create_model_state_filepath(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}.state'.format(self.__model_name))

    # region private methods

    def __get_model_root(self):
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.__experiments_io.get_experiments_dir())

    def __get_model_states_dir(self):

        result_dir = os.path.join(
            self.__get_model_root(),
            os.path.join(u'model_states/'))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion
