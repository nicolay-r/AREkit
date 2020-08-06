from os.path import join
from arekit.common.experiment.data_type import DataType


class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def __init__(self, neutral_root):
        assert(isinstance(neutral_root, unicode))
        self.__get_neutral_root = neutral_root

    def read_neutral_opinion_collection(self, doc_id, data_type):
        raise NotImplementedError()

    def get_doc_ids_set_to_compare(self, doc_ids):
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
        assert(isinstance(data_type, DataType))

        filename = u"art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                           d_type=data_type.name)

        return join(self.__get_neutral_root, filename)
