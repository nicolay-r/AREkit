from arekit.processing.lemmatization.base import Stemmer


class OpinionOperations(object):

    def __init__(self):
        self.__synonyms = None

    # TODO. Already in data.io
    @property
    def SynonymsCollection(self):
        return self.__synonyms

    # TODO. Already in data.io
    def read_synonyms_collection(self, stemmer):
        raise NotImplementedError()

    # TODO. Already in data.io
    def init_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__synonyms = self.read_synonyms_collection(stemmer=stemmer)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        raise NotImplementedError()

    def create_opinion_collection(self):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()
