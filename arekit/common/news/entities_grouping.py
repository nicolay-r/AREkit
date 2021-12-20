from arekit.common.entities.base import Entity
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class EntitiesGroupingPipelineItem(BasePipelineItem):

    def __init__(self, value_to_group_id_func):
        assert(callable(value_to_group_id_func))
        self.__value_to_group_id_func = value_to_group_id_func

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        terms = pipeline_ctx.provide("src")
        assert(isinstance(terms, list))

        for entity in filter(lambda term: isinstance(term, Entity), terms):
            group_index = self.__value_to_group_id_func(entity.Value)
            entity.set_group_index(group_index)
