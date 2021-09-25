import collections

from arekit.common.experiment.input.sample import InputSampleBase
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.base import News


class OpinionProvider(object):
    """
    TextOpinion iterator
    """

    def __init__(self, linked_text_opins_it_func):
        assert(callable(linked_text_opins_it_func))
        self.__linked_text_opins_it_func = linked_text_opins_it_func

    # region private methods

    @staticmethod
    def __iter_linked_text_opinion_lists(news, iter_opins_for_extraction, filter_text_opinion_func):
        assert (isinstance(news, News))
        assert (isinstance(iter_opins_for_extraction, collections.Iterable))
        assert (callable(filter_text_opinion_func))

        for opinion in iter_opins_for_extraction:
            linked_text_opinions = news.extract_linked_text_opinions(opinion)
            assert (linked_text_opinions, LinkedTextOpinionsWrapper)

            filtered_text_opinions = list(filter(filter_text_opinion_func, linked_text_opinions))

            if len(filtered_text_opinions) == 0:
                continue

            yield filtered_text_opinions

    @staticmethod
    def __iter_linked_text_opins(read_news_func, news_opins_for_extraction_func,
                                 parsed_news_it, terms_per_context):
        """
        Extracting text-level opinions based on doc-level opinions in documents,
        obtained by information in experiment.

        NOTE:
        1. Assumes to provide the same label (doc level opinion) onto related text-level opinions.
        """
        assert (callable(read_news_func))
        assert (isinstance(parsed_news_it, collections.Iterable))

        curr_id = 0

        for parsed_news in parsed_news_it:

            linked_text_opinion_lists = OpinionProvider.__iter_linked_text_opinion_lists(
                news=read_news_func(parsed_news.RelatedNewsID),
                iter_opins_for_extraction=news_opins_for_extraction_func(doc_id=parsed_news.RelatedNewsID),
                filter_text_opinion_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                    parsed_news=parsed_news,
                    text_opinion=text_opinion,
                    window_size=terms_per_context))

            for linked_text_opinion_list in linked_text_opinion_lists:

                # Assign IDs.
                for text_opinion in linked_text_opinion_list:
                    text_opinion.set_text_opinion_id(curr_id)
                    curr_id += 1

                yield parsed_news, LinkedTextOpinionsWrapper(linked_text_opinion_list)

    # endregion

    @classmethod
    def create(cls, read_news_func, iter_news_opins_for_extraction,
               parsed_news_it_func, terms_per_context):
        assert(callable(read_news_func))
        assert(isinstance(iter_news_opins_for_extraction, collections.Iterable))
        assert(isinstance(terms_per_context, int))
        assert(callable(parsed_news_it_func))

        def it_func():
            return cls.__iter_linked_text_opins(
                read_news_func=lambda news_id: read_news_func(news_id),
                news_opins_for_extraction_func=lambda news_id: iter_news_opins_for_extraction(doc_id=news_id),
                terms_per_context=terms_per_context,
                parsed_news_it=parsed_news_it_func())

        return cls(linked_text_opins_it_func=it_func)

    def iter_linked_opinion_wrappers(self):
        return self.__linked_text_opins_it_func()
