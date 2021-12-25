import collections

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class FilterPipelineItem(BasePipelineItem):

    def __init__(self, filter_func=None):
        assert(callable(filter_func))
        self.__filter_func = filter_func

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        iter_data = pipeline_ctx.provide("src")
        assert(isinstance(iter_data, collections.Iterable))
        pipeline_ctx.update(param="src", value=filter(self.__filter_func, iter_data))
