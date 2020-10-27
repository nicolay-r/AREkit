

class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    # region abstract methods

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

    def get_doc_ids_set_to_neutrally_annotate(self):
        """ provides set of documents that utilized by neutral annotator algorithm in order to
            provide the related labeling of neutral attitudes in it.
            By default we consider an empty set, so there is no need to ulize neutral annotator.
        """
        raise NotImplementedError()

    def get_doc_ids_set_to_compare(self):
        """ provides a set of document ids, utilized in opinion comparison operation during
            model evaluation process.
        """
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def create_opinion_collection(self, opinions=None):
        raise NotImplementedError()
