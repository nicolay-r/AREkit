class OpinionCollectionsProvider(object):

    def iter_opinions(self, filepath, labels_formatter, error_on_non_supported):
        raise NotImplementedError()

    def serialize(self, collection, filepath, labels_formatter, error_on_non_supported):
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass