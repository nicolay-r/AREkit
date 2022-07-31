from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper


def create_text_parser(exp_ctx, entities_parser, value_to_group_id_func, ppl_items):
    """
    NOTE: For neural networks in terms of `ppl_items` parameter you may adopt the following list:
        [DefaultTextTokenizer(keep_tokens=True),
         LemmasBasedFrameVariantsParser(frame_variants=exp_ctx.FrameVariantCollection,
                                        stemmer=create_stemmer())]
    """
    assert(isinstance(entities_parser, BasePipelineItem))
    assert(callable(value_to_group_id_func) or value_to_group_id_func is None)
    assert(isinstance(ppl_items, list) or ppl_items is None)

    if not isinstance(exp_ctx, ExperimentSerializationContext):
        # We do not utilize text_parser in such case.
        return None

    ppl_entities_grouping = EntitiesGroupingPipelineItem(
        value_to_group_id_func=value_to_group_id_func) \
        if value_to_group_id_func is not None else None

    # We may customize the pipeline after the entities annotation and grouping stages.
    pipeline = [entities_parser, ppl_entities_grouping] + \
               (ppl_items if ppl_items is not None else [])

    return BaseTextParser(pipeline)


def create_stemmer():
    # This is the only stemmer supported by the experiment.
    return MystemWrapper()
