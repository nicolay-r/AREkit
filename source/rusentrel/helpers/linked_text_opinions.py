# -*- coding: utf-8 -*-
from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.opinions.collection import OpinionCollection
from core.common.text_opinions.text_opinion import TextOpinion
from core.source.rusentrel.helpers.context.collection import RuSentRelTextOpinionCollection
from core.source.rusentrel.helpers.context.opinion import RuSentRelTextOpinion
from core.source.rusentrel.news import RuSentRelNews


class RuSentRelNewsTextOpinionExtractorHelper:
    """
    TextOpinion provider from RuSentRel news
    """

    @staticmethod
    def add_entries(text_opinion_collection,
                    rusentrel_news_id,
                    news,
                    opinions,
                    check_text_opinion_is_correct):
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinions, OpinionCollection))
        assert(callable(check_text_opinion_is_correct))

        it_entries = RuSentRelNewsTextOpinionExtractorHelper.__iter_rusentrel_text_opinion_entries(
            rusentrel_news_id=rusentrel_news_id,
            news=news,
            opinions=opinions)

        discarded = 0
        for entries in it_entries:
            assert(isinstance(entries, RuSentRelTextOpinionCollection))

            # TODO. Remove owner
            text_opinions = RuSentRelNewsTextOpinionExtractorHelper.__iter_text_opinions(
                entries=entries,
                owner=text_opinion_collection)

            discarded += text_opinion_collection.add_text_opinions(
                text_opinions=text_opinions,
                check_opinion_correctness=check_text_opinion_is_correct)

        return discarded

    # region private methods

    @staticmethod
    def __iter_text_opinions(entries, owner):
        # TODO. Remove owner
        assert(isinstance(owner, LabeledLinkedTextOpinionCollection))
        for entry in entries:
            yield RuSentRelNewsTextOpinionExtractorHelper.__entry_to_text_opinion(entry=entry, owner=owner)

    @staticmethod
    def __entry_to_text_opinion(entry, owner):
        """
        Text Level Opinion -> Text Opinion
        """
        # TODO. Remove owner
        assert(isinstance(entry, RuSentRelTextOpinion))

        return TextOpinion(
            news_id=entry.RuSentRelNewsId,
            source_id=entry.SourceId,
            target_id=entry.TargetId,
            label=entry.Sentiment,
            owner=owner,
            text_opinion_id=None)

    @staticmethod
    def __iter_rusentrel_text_opinion_entries(rusentrel_news_id, news, opinions):
        """
        Document Level Opinions -> Linked Text Level Opinions
        """
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinions, OpinionCollection))

        def same_sentence_text_opinions(relation):
            return abs(news.Helper.get_sentence_index_by_entity(relation.SourceEntity) -
                       news.Helper.get_sentence_index_by_entity(relation.TargetEntity)) == 0

        for opinion in opinions:

            yield RuSentRelTextOpinionCollection.from_opinion(
                rusentrel_news_id=rusentrel_news_id,
                doc_entities=news.DocEntities,
                opinion=opinion,
                check_text_opinion_correctness=same_sentence_text_opinions)

    # endregion
