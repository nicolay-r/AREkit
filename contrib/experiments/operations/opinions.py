

class OpinionOperations(object):

    def __init__(self):
        self.__synonyms = None

    def read_neutral_opinion_collection(self, doc_id, data_type):
        raise NotImplementedError()

    def create_opinion_collection(self):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def read_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()
