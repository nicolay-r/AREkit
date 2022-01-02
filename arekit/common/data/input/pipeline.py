from arekit.common.data.input.sample import InputSampleBase
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.opinions.base import Opinion
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_flatten import FlattenIterPipelineItem
from arekit.common.pipeline.item_handle import HandleIterPipelineItem
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.text_opinions.base import TextOpinion


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


def text_opinions_iter_pipeline(parse_news_func, iter_doc_opins,
                                value_to_group_id_func, terms_per_context):
    """ Opinion collection generation pipeline.
    """
    assert(callable(parse_news_func))
    assert(callable(iter_doc_opins))
    assert(callable(value_to_group_id_func))
    assert(isinstance(terms_per_context, int))

    def __assign_ids(linkage, curr_id_list):
        assert(isinstance(linkage, TextOpinionsLinkage))
        for text_opinion in linkage:
            assert(isinstance(text_opinion, TextOpinion))
            current_id = curr_id_list[0]
            text_opinion.set_text_opinion_id(current_id)
            curr_id_list[0] += 1

    # List that allows to pass and modify int (current id) into function.
    curr_id = [0]

    return BasePipeline([
        # (id) -> (id, opinions)
        MapPipelineItem(map_func=lambda doc_id: (doc_id, list(iter_doc_opins(doc_id)))),

        # (id, opinions) -> (parsed_news, opinions).
        MapPipelineItem(map_func=lambda data: (parse_news_func(data[0]), data[1])),

        # (parsed_news, opinions) -> (opins_provider, entities_provider, opinions).
        # TODO. #245 adopt DocumentService.
        MapPipelineItem(map_func=lambda data: (
            data[0],
            TextOpinionPairsProvider(parsed_news=data[0], value_to_group_id_func=value_to_group_id_func),
            EntityServiceProvider(parsed_news=data[0]),
            data[1])),

        # (opins_provider, entities_provider, opinions) -> linkages[].
        # TODO. #245 adopt DocumentService.
        MapPipelineItem(map_func=lambda data: __to_text_opinion_linkages(
            provider=data[1],
            opinions=data[3],
            # Assign parsed news.
            tag_value_func=lambda _: data[0],
            filter_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                entity_service=data[2],
                text_opinion=text_opinion,
                window_size=terms_per_context))),

        # linkages[] -> linkages.
        FlattenIterPipelineItem(),

        # Assign id.
        HandleIterPipelineItem(handle_func=lambda linkage: __assign_ids(linkage=linkage, curr_id_list=curr_id))
    ])
