import collections

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class HandleIterPipelineItem(BasePipelineItem):

    def __init__(self, handle_func=None):
        assert(callable(handle_func))
        self.__handle_func = handle_func

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        items_iter = pipeline_ctx.provide("src")
        assert(isinstance(items_iter, collections.Iterable))

        for item in items_iter:
            self.__handle_func(item)
