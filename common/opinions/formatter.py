
# TODO. Move to collection/formatter.
class OpinionCollectionFormatter(object):
    """
    TODO:
        - Remove limitation: saving assumes a certain file per collection. Archieves are not supported.
        - load_from_file depends on synonyms?!
        - labels_formatter?! (also should be removed, however it varies)
    """

    @staticmethod
    def load_from_file(filepath, synonyms, labels_formatter):
        raise NotImplementedError()

    @staticmethod
    def save_to_file(collection, filepath, labels_formatter):
        raise NotImplementedError()


