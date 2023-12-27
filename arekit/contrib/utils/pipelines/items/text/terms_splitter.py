from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces


class TermsSplitterParser(BasePipelineItem):

    def apply_core(self, input_data, pipeline_ctx, **kwargs):
        assert(isinstance(pipeline_ctx, PipelineContext))
        super(TermsSplitterParser, self).apply_core(**kwargs)
        return split_by_whitespaces(input_data)
