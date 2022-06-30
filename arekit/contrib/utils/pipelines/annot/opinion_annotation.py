from arekit.common.data.input.sample import InputSampleBase
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.contrib.utils.pipelines.annot.utils import iter_opinions_as_text_opinion_linkages


def ppl_text_ids_to_parsed_news(parse_news_func):
    assert(callable(parse_news_func))

    return [
        # (id) -> (id, parsed_news).
        MapPipelineItem(map_func=lambda doc_id: (doc_id, parse_news_func(doc_id)))
    ]


def ppl_parsed_to_annotation(annotator, data_type):
    assert(isinstance(annotator, BaseOpinionAnnotator))

    return [
        # (id, parsed_news) -> (id, opinions)
        MapPipelineItem(map_func=lambda data: (
            data[1], annotator.annotate_collection(data_type=data_type, parsed_news=data[1])))
    ]


def ppl_parsed_news_to_opinion_linkages(value_to_group_id_func, terms_per_context, entity_index_func):
    """ Opinion collection generation pipeline.
        NOTE: Here we do not perform IDs assignation!
    """
    assert(callable(value_to_group_id_func))
    assert(isinstance(terms_per_context, int))
    assert(callable(entity_index_func))

    return [

        # (parsed_news, opinions) -> (opins_provider, entities_provider, opinions).
        MapPipelineItem(map_func=lambda data: (
            ParsedNewsService(
                parsed_news=data[0],
                providers=[TextOpinionPairsProvider(value_to_group_id_func=value_to_group_id_func),
                           EntityServiceProvider(entity_index_func=entity_index_func)]),
            data[1])),

        # (opins_provider, entities_provider, opinions) -> linkages[].
        MapPipelineItem(map_func=lambda data: iter_opinions_as_text_opinion_linkages(
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
