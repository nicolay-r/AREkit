from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.parsed import BaseParsedText


class BaseTextParser(BasePipeline):

    def run(self, params_dict, parent_ctx=None):
        assert(isinstance(params_dict, dict))
        ctx = super(BaseTextParser, self).run(pipeline_ctx=PipelineContext(params_dict, parent_ctx=parent_ctx))
        return BaseParsedText(terms=ctx.provide("result"))
