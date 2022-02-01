import collections

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class MapPipelineItem(BasePipelineItem):

    def __init__(self, map_func=None):
        assert(callable(map_func))
        self.__map_func = map_func

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        iter_data = pipeline_ctx.provide("src")
        assert(isinstance(iter_data, collections.Iterable))
        pipeline_ctx.update(param="src", value=map(self.__map_func, iter_data))
