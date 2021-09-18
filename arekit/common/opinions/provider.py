class OpinionCollectionsProvider(object):

    def iter_opinions(self, source, labels_formatter, error_on_non_supported):
        raise NotImplementedError()

    def serialize(self, collection, target, labels_formatter, error_on_non_supported):
        raise NotImplementedError()
