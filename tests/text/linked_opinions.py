from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.text_opinions.base import TextOpinion


def is_same_sentence(text_opinion, parsed_news):
    assert(isinstance(text_opinion, TextOpinion))
    s_ind = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                            position_type=TermPositionTypes.SentenceIndex)
    t_ind = parsed_news.get_entity_position(id_in_document=text_opinion.TargetId,
                                            position_type=TermPositionTypes.SentenceIndex)
    return s_ind == t_ind


def iter_same_sentence_linked_text_opinions(parsed_news, opinions, value_to_group_id_func):
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(opinions, OpinionCollection))

    pairs_provider = TextOpinionPairsProvider(parsed_news=parsed_news,
                                              value_to_group_id_func=value_to_group_id_func)

    for opinion in opinions:

        text_opinions_linkage = TextOpinionsLinkage(
            text_opinions_it=pairs_provider.iter_from_opinion(opinion))

        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))

        if len(text_opinions_linkage) == 0:
            continue

        text_opinion = text_opinions_linkage.First
        assert(isinstance(text_opinion, TextOpinion))
        text_opinion.set_owner(opinions)
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))

        is_same = is_same_sentence(text_opinion=text_opinion, parsed_news=parsed_news)

        if not is_same:
            continue

        yield text_opinion
