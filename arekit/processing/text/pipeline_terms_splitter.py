from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem
from arekit.common.utils import split_by_whitespaces


class TermsSplitterParser(BasePipelineItem):

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        return pipeline_ctx.update(param="src",
                                   value=split_by_whitespaces(pipeline_ctx.provide("src")))
