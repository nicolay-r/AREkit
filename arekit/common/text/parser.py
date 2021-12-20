from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parsed import BaseParsedText


class BaseTextParser(BasePipeline):

    def run(self, pipeline_ctx):
        super(BaseTextParser, self).run(pipeline_ctx)
        return BaseParsedText(terms=pipeline_ctx.provide("src"))
