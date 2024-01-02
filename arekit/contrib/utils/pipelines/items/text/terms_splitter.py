from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces


class TermsSplitterParser(BasePipelineItem):

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        return split_by_whitespaces(input_data)
