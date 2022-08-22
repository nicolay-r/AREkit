from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.annot.algo.base import BaseOpinionAnnotationAlgorithm


class PredefinedOpinionAnnotationAlgorithm(BaseOpinionAnnotationAlgorithm):
    """ A placeholder of the algorithm which is consider to return
        a predefined list of opinions, provided by a given document_id.
    """

    def __init__(self, get_opinions_by_doc_id_func):
        assert(callable(get_opinions_by_doc_id_func))
        self.__get_opinions_by_doc_id_func = get_opinions_by_doc_id_func

    def iter_opinions(self, parsed_news, existed_opinions=None):
        assert(isinstance(parsed_news, ParsedNews))
        return self.__get_opinions_by_doc_id_func(parsed_news.RelatedDocID)
