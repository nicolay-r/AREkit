class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def read_news(self, doc_id):
        raise NotImplementedError()

    def create_parse_options(self):
        raise NotImplementedError()

    def iter_news_indices(self, data_type):
        raise NotImplementedError()