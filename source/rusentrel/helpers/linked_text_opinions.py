# -*- coding: utf-8 -*-
from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.opinions.collection import OpinionCollection
from core.common.opinions.opinion import Opinion
from core.common.text_opinions.text_opinion import TextOpinion
from core.source.rusentrel.helpers.context.collection import RuSentRelTextOpinionCollection
from core.source.rusentrel.helpers.context.opinion import RuSentRelTextOpinion
from core.source.rusentrel.news import RuSentRelNews


# TODO. This should be helper, static helper.
class RuSentRelLinkedTextOpinionCollection(LabeledLinkedTextOpinionCollection):
    """
    TextOpinion provider from RuSentRel news
    """

    def add_rusentrel_news_opinions(self,
                                    rusentrel_news_id,
                                    news,
                                    opinions,
                                    check_text_opinion_is_correct):

        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinions, OpinionCollection))
        assert(callable(check_text_opinion_is_correct))

        it = self.__iter_rusentrel_text_opinion_entries(
            rusentrel_news_id=rusentrel_news_id,
            news=news,
            opinions=opinions)

        total_discarded = 0
        for opinion, text_opinion_entries in it:
            assert(isinstance(opinion, Opinion))
            assert(isinstance(text_opinion_entries, RuSentRelTextOpinionCollection))

            total_discarded += self.__add_rusentrel_text_opinion(
                text_opinion_entries=text_opinion_entries,
                label=opinion.Sentiment,    # TODO. Provide custom func for label generation
                check_text_opinion_is_correct=check_text_opinion_is_correct)

        return total_discarded

    # region private methods

    def __add_rusentrel_text_opinion(self, text_opinion_entries, label, check_text_opinion_is_correct):
        assert(isinstance(text_opinion_entries, RuSentRelTextOpinionCollection))
        assert(callable(check_text_opinion_is_correct))

        discarded = 0
        for index, rusentrel_text_opinion in enumerate(text_opinion_entries):
            assert(isinstance(rusentrel_text_opinion, RuSentRelTextOpinion))

            text_opinion = TextOpinion(
                news_id=rusentrel_text_opinion.RuSentRelNewsId,
                text_opinion_id=len(self),
                source_id=rusentrel_text_opinion.SourceId,
                target_id=rusentrel_text_opinion.TargetId,
                owner=self,
                label=label)

            if not check_text_opinion_is_correct(text_opinion):
                discarded += 1
                continue

            self.register_text_opinion(text_opinion)

        self.set_none_for_last_text_opinion()
        return discarded

    @staticmethod
    def __iter_rusentrel_text_opinion_entries(rusentrel_news_id, news, opinions):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinions, OpinionCollection))

        def same_sentence_text_opinions(relation):
            return abs(news.Helper.get_sentence_index_by_entity(relation.SourceEntity) -
                       news.Helper.get_sentence_index_by_entity(relation.TargetEntity)) == 0

        for opinion in opinions:

            entries = RuSentRelTextOpinionCollection.from_opinion(
                rusentrel_news_id=rusentrel_news_id,
                doc_entities=news.DocEntities,
                opinion=opinion,
                check_text_opinion_correctness=same_sentence_text_opinions)

            yield opinion, entries

    # endregion
