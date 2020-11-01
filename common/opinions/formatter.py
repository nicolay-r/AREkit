
class OpinionCollectionsFormatter(object):

    def iter_opinions_from_file(self, filepath, labels_formatter):
        raise NotImplementedError()

    def save_to_file(self, collection, filepath, labels_formatter):
        raise NotImplementedError()

    def save_to_archive(self, collections_iter, labels_formatter):
        """
        collections_iter: iterator of pairs
            enumeration of pairs (news_id, collection)
        synonyms: SynonymsCollection
        labels_formatter: LabelsFormatter
        """
        raise NotImplementedError()