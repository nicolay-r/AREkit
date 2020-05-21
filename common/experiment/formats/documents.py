from arekit.common.frame_variants.collection import FrameVariantsCollection


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

    def iter_parsed_news(self, data_type, frame_variant_collection):
        assert(isinstance(frame_variant_collection, FrameVariantsCollection))
        for doc_id in self.iter_news_indices(data_type):
            news = self.read_news(doc_id=doc_id)
            yield news.parse(options=self.create_parse_options(),
                             frame_variant_collection=frame_variant_collection)
