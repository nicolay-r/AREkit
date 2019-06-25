# -*- coding: utf-8 -*-
from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.linked_text_opinions.text_opinion import TextOpinion
from core.source.rusentrel.helpers.context.collection import RuSentRelContextOpinionList
from core.source.rusentrel.helpers.context.opinion import RuSentRelContextOpinion


class RuSentRelLinkedTextOpinionCollection(LabeledLinkedTextOpinionCollection):

    NO_NEXT_RELATION = None

    def add_rusentrel_context_opinons(
            self,
            # TODO. Generate ContextOpinionList from opinion here (in this class).
            context_opinions,
            label,
            rusentrel_news_id,
            check_text_opinion_is_correct):
        assert(isinstance(context_opinions, RuSentRelContextOpinionList))
        assert(isinstance(rusentrel_news_id, int))
        assert(callable(check_text_opinion_is_correct))

        missed = 0
        for index, opinion in enumerate(context_opinions):
            assert(isinstance(opinion, RuSentRelContextOpinion))

            text_opinion = TextOpinion(
                news_id=rusentrel_news_id,
                text_opinion_id=self.TextOpinionCount,
                source_id=opinion.SourceEntity.IdInDocument,
                target_id=opinion.TargetEntity.IdInDocument,
                owner=self,
                label=label)

            if not check_text_opinion_is_correct(text_opinion):
                missed += 1
                continue

            self.register_text_opinion(text_opinion)

        self.set_none_for_last_text_opinion()
        return missed
