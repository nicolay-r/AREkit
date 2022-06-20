from arekit.common.data.input.sample import InputSampleBase
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.opinions.annot.base import BaseAnnotator
from arekit.common.opinions.base import Opinion
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.text_opinions.base import TextOpinion


def ppl_text_ids_to_parsed_news(parse_news_func):
    assert(callable(parse_news_func))

    return [
        # (id) -> (id, parsed_news).
        MapPipelineItem(map_func=lambda doc_id: (doc_id, parse_news_func(doc_id)))
    ]


def ppl_parsed_to_annotation(annotator, data_type):
    assert(isinstance(annotator, BaseAnnotator))

    return [
        # (id, parsed_news) -> (id, opinions)
        MapPipelineItem(map_func=lambda data: (
            data[0], annotator.annotate_collection(data_type=data_type, parsed_news=data[1])))
    ]


def ppl_parsed_news_to_opinion_linkages(value_to_group_id_func, terms_per_context):
    """ Opinion collection generation pipeline.
        NOTE: Here we do not perform IDs assignation!
    """
    assert(callable(value_to_group_id_func))
    assert(isinstance(terms_per_context, int))

    return [

        # (parsed_news, opinions) -> (opins_provider, entities_provider, opinions).
        MapPipelineItem(map_func=lambda data: (
            ParsedNewsService(
                parsed_news=data[0],
                providers=[TextOpinionPairsProvider(value_to_group_id_func=value_to_group_id_func),
                           EntityServiceProvider()]),
            data[1])),

        # (opins_provider, entities_provider, opinions) -> linkages[].
        MapPipelineItem(map_func=lambda data: __to_text_opinion_linkages(
            provider=data[0].get_provider(TextOpinionPairsProvider.NAME),
            opinions=data[1],
            # Assign parsed news.
            tag_value_func=lambda _: data[0],   # ParsedNewsService
            filter_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                entity_service=data[0].get_provider(EntityServiceProvider.NAME),
                text_opinion=text_opinion,
                window_size=terms_per_context))),

        # linkages[] -> linkages.
        FlattenIterPipelineItem()
    ]


def __to_text_opinion_linkages(provider, opinions, tag_value_func, filter_func):
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

        linkage = TextOpinionsLinkage(text_opinions)

        if tag_value_func is not None:
            linkage.set_tag(tag_value_func(linkage))

        yield linkage
