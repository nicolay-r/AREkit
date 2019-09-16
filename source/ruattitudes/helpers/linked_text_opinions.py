from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.opinions.collection import OpinionCollection
from core.common.ref_opinon import RefOpinion
from core.common.text_opinions.text_opinion import TextOpinion
from core.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from core.source.ruattitudes.news import RuAttitudesNews
from core.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNewsTextOpinionExtractorHelper:
    """
    TextOpinion provider from RuAttitudesNews
    """

    @staticmethod
    def add_entries(text_opinion_collection,
                    news,
                    opinions,
                    check_text_opinion_is_correct):
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(news, RuAttitudesNews))
        assert(isinstance(opinions, OpinionCollection))
        assert(callable(check_text_opinion_is_correct))

        discarded = 0
        for opinion, sentences in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news):

            text_opinions = RuAttitudesNewsTextOpinionExtractorHelper.__iter_text_opinions(
                opinion=opinion,
                sentences=sentences)

            discarded += text_opinion_collection.try_add_text_opinions(
                text_opinions=text_opinions,
                check_opinion_correctness=check_text_opinion_is_correct)

        return discarded

    # region private methods

    @staticmethod
    def __iter_text_opinions(opinion, sentences):
        for sentence in sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            ref_opinion = sentence.find_ref_opinion_by_key(key=opinion.Tag)
            yield RuAttitudesNewsTextOpinionExtractorHelper.__ref_opinion_to_text_opinion(
                news_index=sentence.Owner.NewsIndex,
                ref_opinion=ref_opinion)

    @staticmethod
    def __ref_opinion_to_text_opinion(news_index, ref_opinion):
        assert(isinstance(news_index, int))
        assert(isinstance(ref_opinion, RefOpinion))

        cloned_ref_opinion = RefOpinion(
            source_id=ref_opinion.SourceId,
            target_id=ref_opinion.TargetId,
            sentiment=ref_opinion.Sentiment)

        return TextOpinion.create_from_ref_opinion(
            news_id=news_index,
            text_opinion_id=None,
            ref_opinion=cloned_ref_opinion)

    # endregion
