import os
from os import path

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIOUtils
from arekit.contrib.experiments.nn_io.utils import rm_dir_contents
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.networks.nn_io import NeuralNetworkIO
from arekit.processing.lemmatization.base import Stemmer


class BaseExperimentNeuralNetworkIO(NeuralNetworkIO):

    def __init__(self, experiments_io, model_name):
        assert(isinstance(experiments_io, BaseExperimentsIOUtils))
        self.__experiments_io = experiments_io
        self.__model_name = model_name
        self.__synonyms = None

    # region implemented nn_io

    @property
    def ModelSavePathPrefix(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}'.format(self.__model_name))

    def __get_model_states_dir(self):

        result_dir = os.path.join(
            self.__get_model_root(),
            os.path.join(u'model_states/'))

        create_dir_if_not_exists(result_dir)
        return result_dir

    def __get_model_root(self):
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.__experiments_io.get_experiments_dir())

    # endregion

    # region 'get' public methods

    @property
    def ExperimentsIO(self):
        return self.__experiments_io

    # endregion

    # region 'write' methods

    def write_log(self, log_names, log_values):
        assert(isinstance(log_names, list))
        assert(isinstance(log_values, list))
        assert(len(log_names) == len(log_values))

        log_path = os.path.join(self.__get_model_root(), u"log.txt")

        with open(log_path, 'w') as f:
            for index, log_value in enumerate(log_values):
                f.write("{}: {}\n".format(log_names[index], log_value))

    @staticmethod
    def prepare_model_root(model_root, rm_contents=True):
        if not rm_contents:
            return

        rm_dir_contents(model_root)

    def get_logfile_dir(self):
        return path.join(self.__get_model_root(), u"log/")

    # TODO. Everything below is a part of BaseExperimentDataProcessing

    @property
    def SynonymsCollection(self):
        return self.__synonyms

    def iter_doc_ids(self, data_type):
        raise NotImplementedError()

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):
        raise NotImplementedError()

    def read_neutral_opinion_collection(self, doc_id, data_type):
        raise NotImplementedError()

    def read_synonyms_collection(self, stemmer):
        raise NotImplementedError()

    def create_opinion_collection(self):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def init_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__synonyms = self.read_synonyms_collection(stemmer=stemmer)
