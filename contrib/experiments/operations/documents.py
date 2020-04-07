class DocumentOperations(object):

    def read_parsed_news(self, doc_id):
        raise NotImplementedError()

    def iter_news_indices(self, data_type):
        raise NotImplementedError()