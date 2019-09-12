from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.opinions.collection import OpinionCollection
from core.source.ruattitudes.helpers.news_helper import NewsHelper
from core.source.ruattitudes.news import RuAttitudesNews


class RuAttitudesNewsTextOpinionExtractorHelper:
    """
    TextOpinion provider from RuAttitudesNews
    """

    @staticmethod
    def add_entries(text_opinion_collection,
                    news_id,
                    news,
                    opinions,
                    check_text_opinion_is_correct):
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(news_id, int))
        assert(isinstance(news, RuAttitudesNews))
        assert(isinstance(opinions, OpinionCollection))
        assert(callable(check_text_opinion_is_correct))

        for opinion, sentences in NewsHelper.iter_opinions_with_related_sentences(news):
            pass
