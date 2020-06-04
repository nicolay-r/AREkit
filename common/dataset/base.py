from arekit.common.dataset.doc_ind_mapper import DocumentIndexMapper
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.collection import TextOpinionCollection


class Dataset(object):
    """
    Provides an aggregation of multiple sources.
    """

    def __init__(self, parsed_doc_collection, text_opinion_collection, doc_ind_mapper):
        assert(isinstance(parsed_doc_collection, ParsedNewsCollection))
        assert(isinstance(text_opinion_collection, TextOpinionCollection))
        assert(isinstance(doc_ind_mapper, DocumentIndexMapper))

        self.__parsed_doc_collection = parsed_doc_collection
        self.__text_opinion_collection = text_opinion_collection
        self.__doc_ind_mapper = doc_ind_mapper
