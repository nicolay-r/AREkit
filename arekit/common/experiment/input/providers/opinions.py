import collections

from arekit.common.experiment.extract.opinions import iter_linked_text_opins


class OpinionProvider(object):
    """
    TextOpinion iterator
    """

    def __init__(self, linked_text_opins_it_func):
        assert(callable(linked_text_opins_it_func))
        self.__linked_text_opins_it_func = linked_text_opins_it_func

    @classmethod
    def create(cls, read_news_func, iter_news_opins_for_extraction,
               parsed_news_it_func, terms_per_context):
        assert(callable(read_news_func))
        assert(isinstance(iter_news_opins_for_extraction, collections.Iterable))
        assert(isinstance(terms_per_context, int))
        assert(callable(parsed_news_it_func))

        def it_func():
            return iter_linked_text_opins(
                read_news_func=lambda news_id: read_news_func(news_id),
                news_opins_for_extraction_func=lambda news_id: iter_news_opins_for_extraction(doc_id=news_id),
                terms_per_context=terms_per_context,
                parsed_news_it=parsed_news_it_func())

        return cls(linked_text_opins_it_func=it_func)

    def iter_linked_opinion_wrappers(self):
        for data in self.__linked_text_opins_it_func():
            yield data
