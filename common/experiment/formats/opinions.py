from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.utils import create_dir_if_not_exists


class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def __init__(self, results_dir, annot_name_func):
        assert(isinstance(results_dir, unicode))
        assert(callable(annot_name_func))
        self.__results_dir = results_dir
        self.__annot_name_func = annot_name_func

    # region private methods

    @staticmethod
    def __data_type_to_string(data_type):
        if data_type == DataType.Train:
            return u'train'
        if data_type == DataType.Test:
            return u'test'

    # endregion

    def read_neutral_opinion_collection(self, doc_id, data_type):
        raise NotImplementedError()

    def iter_doc_ids_to_compare(self, doc_ids):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def read_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()

    def create_opinion_collection(self, opinions=None):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def create_neutral_opinion_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        results_dir = self.__results_dir

        root = get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__annot_name_func(),
            experiments_dir=results_dir)

        create_dir_if_not_exists(root)

        filename = u"art{doc_id}.neut.{d_type}.txt".format(
            doc_id=doc_id,
            d_type=self.__data_type_to_string(data_type))

        return join(root, filename)
