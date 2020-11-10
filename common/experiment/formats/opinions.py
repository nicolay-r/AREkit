from arekit.common.synonyms import SynonymsCollection


class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def __init__(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        self.__synonyms = synonyms

    @property
    def SynonymsCollection(self):
        """ Synonyms collection utlized for named entities matching.
        """
        return self.__synonyms

    def try_read_neutral_opinion_collection(self, doc_id, data_type):
        """ data_type denotes a set of neutral opinions, where in case of 'train' these are
            opinions that were ADDITIONALLY found to sentiment, while for 'train' these are
            all the opinions that could be found in document.
        """
        raise NotImplementedError()

    def iter_opinions_for_extraction(self, doc_id, data_type):
        """ providing opinions for further context-level opinion extraction process.
            in terms of sentiment attitude extraction, this is a general method
            which provides all the possible opinions within a particular document.
        """
        raise NotImplementedError()

    def save_neutral_opinion_collection(self, collection, labels_fmt, doc_id, data_type):
        raise NotImplementedError()

    def read_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()

    def read_result_opinion_collection(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def create_opinion_collection(self, opinions=None):
        raise NotImplementedError()
