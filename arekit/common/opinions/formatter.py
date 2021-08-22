# TODO. Rename this into OpinionCollectionProvider. #175.
# TODO. Utilize it in `with` block!!! #175.
class OpinionCollectionsFormatter(object):

    # TODO. rename as iter_opinions(self) #175.
    def iter_opinions_from_file(self, filepath, labels_formatter, error_on_non_supported):
        raise NotImplementedError()

    # TODO. rename as serialize(collection)#175.
    def save_to_file(self, collection, filepath, labels_formatter, error_on_non_supported):
        raise NotImplementedError()

    # TODO. In nested provider! (Remove from here) #175
    # TODO. In nested provider! (Remove from here) #175
    # TODO. In nested provider! (Remove from here) #175
    def save_to_archive(self, collections_iter, labels_formatter):
        """
        collections_iter: iterator of pairs
            enumeration of pairs (news_id, collection)
        synonyms: SynonymsCollection
        labels_formatter: LabelsFormatter
        """
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass