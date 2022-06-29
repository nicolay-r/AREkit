from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion


def iter_opinions_as_text_opinion_linkages(provider, opinions, tag_value_func, filter_func):
    """ Convert opinion to text_opinions and then build them up into linkages.
    """
    assert(isinstance(provider, TextOpinionPairsProvider))
    assert(callable(tag_value_func))
    assert(callable(filter_func))

    for opinion in opinions:
        assert(isinstance(opinion, Opinion))

        text_opinions = []

        for text_opinion in provider.iter_from_opinion(opinion):
            assert(isinstance(text_opinion, TextOpinion))

            if not filter_func(text_opinion):
                continue

            text_opinions.append(text_opinion)

        if len(text_opinions) == 0:
            continue

        text_opinion_linkage = TextOpinionsLinkage(text_opinions)

        if tag_value_func is not None:
            text_opinion_linkage.set_tag(tag_value_func(text_opinion_linkage))

        yield text_opinion_linkage
