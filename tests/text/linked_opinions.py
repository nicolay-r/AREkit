from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.base import News
from arekit.common.news.parsed.base import ParsedNews
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


def iter_same_sentence_linked_text_opinions(news, parsed_news, opinions):
    assert(isinstance(news, News))
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(opinions, OpinionCollection))

    for opinion in opinions:
        text_opinion_wrap = news.extract_linked_text_opinions(opinion)

        assert(isinstance(text_opinion_wrap, LinkedTextOpinionsWrapper))

        if len(text_opinion_wrap) == 0:
            continue

        text_opinion = text_opinion_wrap.First
        assert(isinstance(text_opinion, TextOpinion))
        text_opinion.set_owner(opinions)
        assert(isinstance(text_opinion_wrap, LinkedTextOpinionsWrapper))

        is_same = is_same_sentence(text_opinion=text_opinion, parsed_news=parsed_news)

        if not is_same:
            continue

        yield text_opinion
