
class OpinionCollectionFormatter(object):

    @staticmethod
    def load_from_file(filepath, synonyms, labels_formatter):
        raise NotImplementedError()

    @staticmethod
    def save_to_file(collection, filepath, labels_formatter):
        raise NotImplementedError()


