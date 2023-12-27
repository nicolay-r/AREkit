from arekit.common.context.token import Token
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.docs.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.docs.parsed.term_position import TermPositionTypes
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.frames.variants.base import FrameVariant
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.text_opinions.base import TextOpinion


def pad_right_inplace(lst, pad_size, filler):

    assert(pad_size - len(lst) > 0)
    lst.extend([filler] * (pad_size - len(lst)))


def in_window(window_begin, window_end, ind):
    return window_begin <= ind < window_end


def __is_same_sentence(text_opinion, entity_service):
    assert(isinstance(text_opinion, TextOpinion))

    s_ind = entity_service.get_entity_position(id_in_document=text_opinion.SourceId,
                                               position_type=TermPositionTypes.SentenceIndex)
    t_ind = entity_service.get_entity_position(id_in_document=text_opinion.TargetId,
                                               position_type=TermPositionTypes.SentenceIndex)
    return s_ind == t_ind


def iter_same_sentence_linked_text_opinions(pairs_provider, entity_service, opinions):
    assert(isinstance(pairs_provider, TextOpinionPairsProvider))
    assert(isinstance(entity_service, EntityServiceProvider))
    assert(isinstance(opinions, OpinionCollection))

    for opinion in opinions:

        text_opinions_linkage = TextOpinionsLinkage(
            linked_data=pairs_provider.iter_from_opinion(opinion))

        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))

        if len(text_opinions_linkage) == 0:
            continue

        text_opinion = text_opinions_linkage.First
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))

        is_same = __is_same_sentence(text_opinion=text_opinion, entity_service=entity_service)

        if not is_same:
            continue

        yield text_opinion


def terms_to_str(terms):
    r = []
    for t in terms:
        if isinstance(t, str):
            r.append(t)
        elif isinstance(t, Token):
            r.append(t.get_token_value())
        elif isinstance(t, Entity):
            r.append("[{}]".format(t.Value))
        elif isinstance(t, TextFrameVariant):
            r.append("<{}>".format(t.Variant.get_value()))
        elif isinstance(t, FrameVariant):
            r.append(t.get_value())
        else:
            r.append(t)

    return r
