import collections

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class HandleIterPipelineItem(BasePipelineItem):

    def __init__(self, handle_func=None):
        assert(callable(handle_func))
        self.__handle_func = handle_func

    def __updated_data(self, items_iter):
        for item in items_iter:
            # Perform item handling
            self.__handle_func(item)
            yield item

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        items_iter = pipeline_ctx.provide("src")
        assert(isinstance(items_iter, collections.Iterable))
        pipeline_ctx.update("src", value=self.__updated_data(items_iter))
