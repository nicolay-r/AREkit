

class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def try_read_neutral_opinion_collection(self, doc_id, data_type):
        """ data_type denotes a set of neutral opinions, where in case of 'train' these are
            opinions that were ADDITIONALLY found to sentiment, while for 'train' these are
            all the opinions that could be found in document.
        """
        raise NotImplementedError()

    def save_neutral_opinion_collection(self, collection, labels_fmt):
        raise NotImplementedError()

    def read_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def create_opinion_collection(self, opinions=None):
        raise NotImplementedError()
