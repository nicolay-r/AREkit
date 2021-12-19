import collections

from arekit.common.data.input.sample import InputSampleBase
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider


class OpinionProvider(object):
    """
    TextOpinion iterator.
        - Filter text_opinions by provided func.
        - Assigns the IDs.
    """

    def __init__(self, text_opinions_linkages_it_func):
        assert(callable(text_opinions_linkages_it_func))
        self.__text_opinions_linkages_it_func = text_opinions_linkages_it_func

    # region private methods

    @staticmethod
    def __iter_linked_text_opinion_lists(
            text_opinion_pairs_provider,
            iter_opins_for_extraction,
            filter_text_opinion_func):

        assert (isinstance(text_opinion_pairs_provider, TextOpinionPairsProvider))
        assert (isinstance(iter_opins_for_extraction, collections.Iterable))
        assert (callable(filter_text_opinion_func))

        for opinion in iter_opins_for_extraction:
            linked_text_opinions = TextOpinionsLinkage(text_opinion_pairs_provider.iter_from_opinion(opinion))
            filtered_text_opinions = list(filter(filter_text_opinion_func, linked_text_opinions))

            if len(filtered_text_opinions) == 0:
                continue

            yield filtered_text_opinions

    @staticmethod
    def __iter_linked_text_opins(news_opins_for_extraction_func, parse_news_func,
                                 value_to_group_id_func, terms_per_context, doc_ids_it):
        """
        Extracting text-level opinions based on doc-level opinions in documents,
        obtained by information in experiment.

        NOTE:
        1. Assumes to provide the same label (doc level opinion) onto related text-level opinions.
        """
        assert(callable(parse_news_func))
        assert(callable(value_to_group_id_func))
        assert(isinstance(doc_ids_it, collections.Iterable))

        curr_id = 0

        value_to_group_id_func = value_to_group_id_func

        for doc_id in doc_ids_it:

            parsed_news = parse_news_func(doc_id)

            linked_text_opinion_lists = OpinionProvider.__iter_linked_text_opinion_lists(
                text_opinion_pairs_provider=TextOpinionPairsProvider(
                    parsed_news=parsed_news,
                    value_to_group_id_func=value_to_group_id_func),
                iter_opins_for_extraction=news_opins_for_extraction_func(doc_id=parsed_news.RelatedDocID),
                filter_text_opinion_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                    parsed_news=parsed_news,
                    text_opinion=text_opinion,
                    window_size=terms_per_context))

            for linked_text_opinion_list in linked_text_opinion_lists:

                # Assign IDs.
                for text_opinion in linked_text_opinion_list:
                    text_opinion.set_text_opinion_id(curr_id)
                    curr_id += 1

                yield parsed_news, TextOpinionsLinkage(linked_text_opinion_list)

    # endregion

    @classmethod
    def create(cls, iter_news_opins_for_extraction, value_to_group_id_func,
               parse_news_func, terms_per_context):
        assert(callable(iter_news_opins_for_extraction))
        assert(callable(value_to_group_id_func))
        assert(isinstance(terms_per_context, int))
        assert(callable(parse_news_func))

        def it_func(doc_ids_it):
            return cls.__iter_linked_text_opins(
                value_to_group_id_func=value_to_group_id_func,
                news_opins_for_extraction_func=lambda doc_id: iter_news_opins_for_extraction(doc_id=doc_id),
                terms_per_context=terms_per_context,
                doc_ids_it=doc_ids_it,
                parse_news_func=lambda doc_id: parse_news_func(doc_id))

        return cls(text_opinions_linkages_it_func=it_func)

    def iter_linked_opinions(self, doc_ids_it):
        return self.__text_opinions_linkages_it_func(doc_ids_it)
