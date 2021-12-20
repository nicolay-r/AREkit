from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.pipeline.item import BasePipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer


def create_text_parser(exp_data, entities_parser, value_to_group_id_func):
    assert(isinstance(entities_parser, BasePipelineItem))
    assert(callable(value_to_group_id_func) or value_to_group_id_func is None)

    if not isinstance(exp_data, SerializationData):
        # We do not utlize text_parser in such case.
        return None

    ppl_entities_grouping = EntitiesGroupingPipelineItem(
        value_to_group_id_func=value_to_group_id_func) \
        if value_to_group_id_func is not None else None

    pipeline = [entities_parser,
                ppl_entities_grouping,
                DefaultTextTokenizer(keep_tokens=True),
                LemmasBasedFrameVariantsParser(frame_variants=exp_data.FrameVariantCollection,
                                               stemmer=exp_data.Stemmer)]

    return BaseTextParser(pipeline)
