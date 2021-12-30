from arekit.common.data.input.sample import InputSampleBase
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.opinions.base import Opinion
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_flatten import FlattenIterPipelineItem
from arekit.common.pipeline.item_handle import HandleIterPipelineItem
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.text_opinions.base import TextOpinion


def to_text_opinions_iter(provider, opinions, filter_func):
    assert(isinstance(provider, TextOpinionPairsProvider))

    for opinion in opinions:
        assert(isinstance(opinion, Opinion))
        for text_opinion in provider.iter_from_opinion(opinion):
            if not filter_func(text_opinion):
                continue
            yield text_opinion


def process_input_text_opinions(parse_news_func, value_to_group_id_func, terms_per_context):
    """ Opinion collection generation pipeline.
    """

    def __assign_ids(text_opinion, curr_id_list):
        assert(isinstance(text_opinion, TextOpinion))
        current_id = curr_id_list[0]
        text_opinion.set_text_opinion_id(current_id)
        curr_id_list[0] += 1

    # List that allows to pass and modify int (current id) into function.
    curr_id = [0]

    return BasePipeline([

        # (id, opinions) -> (parsed_news, opinions).
        MapPipelineItem(map_func=lambda data: (parse_news_func(data[0]), data[1]) ),

        # (parsed_news, opinions) -> (opins_provider, entities_provider, opinions).
        MapPipelineItem(map_func=lambda data: (
            TextOpinionPairsProvider(parsed_news=data[0], value_to_group_id_func=value_to_group_id_func),
            EntityServiceProvider(parsed_news=data[0]),
            data[1])),

        # (opins_provider, entities_provider, opinions) -> text_opinions[].
        MapPipelineItem(map_func=lambda data: to_text_opinions_iter(
            provider=data[0],
            opinions=data[2],
            filter_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                entity_service=data[1],
                text_opinion=text_opinion,
                window_size=terms_per_context))),

        # text_opinions[] -> text_opinions.
        FlattenIterPipelineItem(),

        # Assign id.
        HandleIterPipelineItem(handle_func=lambda text_opinion: __assign_ids(text_opinion=text_opinion,
                                                                             curr_id_list=curr_id))
    ])