from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.opinions.collection import OpinionCollection
from core.common.ref_opinon import RefOpinion
from core.common.text_opinions.text_opinion import TextOpinion
from core.source.ruattitudes.helpers.news_helper import NewsHelper
from core.source.ruattitudes.news import RuAttitudesNews
from core.source.ruattitudes.sentence import RuAttitudesSentence


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

        discarded = 0
        for opinion, sentences in NewsHelper.iter_opinions_with_related_sentences(news):

            text_opinions = RuAttitudesNewsTextOpinionExtractorHelper.__iter_text_opinions(
                opinion=opinion,
                sentences=sentences)

            discarded += text_opinion_collection.add_text_opinions(
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
                owner=sentence.Owner,
                ref_opinion=ref_opinion)

    @staticmethod
    def __ref_opinion_to_text_opinion(owner, ref_opinion):
        # TODO. Owner should be text_opinion_collection type.
        # TODO. And defined not here
        # TODO. Remove related parameter
        assert(isinstance(owner, RuAttitudesNews))
        assert(isinstance(ref_opinion, RefOpinion))

        # TODO. create new instance of ref_opinion without owner!

        return TextOpinion.from_ref_opinion(
            news_id=owner.NewsIndex,
            text_opinion_id=None,
            ref_opinion=ref_opinion)

    # endregion
