import collections

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class FlattenIterPipelineItem(BasePipelineItem):
    """ Considered to flat iterations of items that represent iterations.
    """

    def __flat_iter(self, iter_data):
        for iter_item in iter_data:
            for item in iter_item:
                yield item

    def apply(self, pipeline_ctx):
        assert (isinstance(pipeline_ctx, PipelineContext))
        iter_data = pipeline_ctx.provide("src")
        assert (isinstance(iter_data, collections.Iterable))
        pipeline_ctx.update(param="src", value=self.__flat_iter(iter_data))
