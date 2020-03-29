from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.text_opinions.base import RefOpinion
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNewsTextOpinionExtractorHelper:
    """
    TextOpinion provider from RuAttitudesNews
    """

    # TODO. Duplicated in RuSentRel
    # TODO. Should be iterator of text_opinions.
    # TODO. Rename as "extract_text_opinions"
    @staticmethod
    def add_entries(text_opinion_collection,
                    news,
                    check_text_opinion_is_correct):
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(news, RuAttitudesNews))
        assert(callable(check_text_opinion_is_correct))

        discarded = 0
        for opinion, sentences in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news):

            text_opinions = RuAttitudesNewsTextOpinionExtractorHelper.__iter_all_text_opinions_in_sentences(
                opinion=opinion,
                sentences=sentences)

            discarded += text_opinion_collection.try_add_linked_text_opinions(
                linked_text_opinions=text_opinions,
                check_opinion_correctness=check_text_opinion_is_correct)

        return discarded

    # region private methods

    # TODO. This should be public.
    @staticmethod
    def __iter_all_text_opinions_in_sentences(opinion, sentences):
        for sentence in sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            ref_opinion = sentence.find_ref_opinion_by_key(key=opinion.Tag)
            yield RuAttitudesNewsTextOpinionExtractorHelper.__ref_opinion_to_text_opinion(
                news_index=sentence.Owner.NewsIndex,
                ref_opinion=ref_opinion,
                sent_to_doc_id_func=sentence.get_doc_level_text_object_id)

    @staticmethod
    def __ref_opinion_to_text_opinion(news_index,
                                      ref_opinion,
                                      sent_to_doc_id_func):
        assert(isinstance(news_index, int))
        assert(isinstance(ref_opinion, RefOpinion))
        assert(callable(sent_to_doc_id_func))

        cloned_ref_opinion = RefOpinion(
            source_id=sent_to_doc_id_func(ref_opinion.SourceId),
            target_id=sent_to_doc_id_func(ref_opinion.TargetId),
            sentiment=ref_opinion.Sentiment)

        return TextOpinion.create_from_ref_opinion(
            news_id=news_index,
            text_opinion_id=None,
            ref_opinion=cloned_ref_opinion)

    # endregion
