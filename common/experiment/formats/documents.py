class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def iter_suppoted_data_types(self):
        raise NotImplementedError()

    def read_news(self, doc_id):
        raise NotImplementedError()

    def create_parse_options(self):
        raise NotImplementedError()

    def iter_news_indices(self, data_type):
        raise NotImplementedError()

    def iter_parsed_news(self, doc_inds, frame_variant_collection):
        for doc_id in doc_inds:
            yield self.__parse_news(doc_id=doc_id,
                                    frame_variant_collection=frame_variant_collection)

    def __parse_news(self, doc_id, frame_variant_collection):
        news = self.read_news(doc_id=doc_id)
        return news.parse(options=self.create_parse_options(),
                          frame_variant_collection=frame_variant_collection)
