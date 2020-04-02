# -*- coding: utf-8 -*-
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.source.rusentrel.helpers.context.collection import RuSentRelTextOpinionCollection
from arekit.source.rusentrel.helpers.context.opinion import RuSentRelTextOpinion
from arekit.source.rusentrel.news import RuSentRelNews


class RuSentRelNewsTextOpinionExtractorHelper:
    """
    TextOpinion provider from RuSentRel news
    """

    @staticmethod
    def iter_text_opinions(news, opinions):
        assert(isinstance(news, RuSentRelNews))
        # TODO: opinions in document + Neutral (optional). Weird
        assert(isinstance(opinions, OpinionCollection))

        it_entries = RuSentRelNewsTextOpinionExtractorHelper.__iter_rusentrel_text_opinion_entries(
            news=news,
            opinions=opinions)

        for entries in it_entries:
            assert(isinstance(entries, RuSentRelTextOpinionCollection))

            text_opinions = RuSentRelNewsTextOpinionExtractorHelper.__iter_text_opinions(entries=entries)

            for text_opinion in text_opinions:
                yield text_opinion

    # region private methods

    # TODO. This should be public.
    @staticmethod
    def __iter_text_opinions(entries):
        for entry in entries:
            yield RuSentRelNewsTextOpinionExtractorHelper.__entry_to_text_opinion(entry=entry)

    @staticmethod
    def __entry_to_text_opinion(entry):
        """
        Text Level Opinion -> Text Opinion
        """
        assert(isinstance(entry, RuSentRelTextOpinion))

        return TextOpinion(
            news_id=entry.RuSentRelNewsId,
            source_id=entry.SourceId,
            target_id=entry.TargetId,
            label=entry.Sentiment,
            owner=None,
            text_opinion_id=None)

    @staticmethod
    def __iter_rusentrel_text_opinion_entries(news, opinions):
        """
        Document Level Opinions -> Linked Text Level Opinions
        """
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinions, OpinionCollection))

        def same_sentence_text_opinions(opinion):
            return abs(news.Helper.get_sentence_index_by_entity(opinion.SourceEntity) -
                       news.Helper.get_sentence_index_by_entity(opinion.TargetEntity)) == 0

        for opinion in opinions:

            yield RuSentRelTextOpinionCollection.from_opinion(
                rusentrel_news_id=news.DocumentID,
                doc_entities=news.DocEntities,
                opinion=opinion,
                check_text_opinion_correctness=same_sentence_text_opinions)

    # endregion
