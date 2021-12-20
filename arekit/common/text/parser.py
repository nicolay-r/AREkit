from arekit.common.text.parsed import BaseParsedText
from arekit.common.text.pipeline_ctx import PipelineContext
from arekit.common.text.pipeline_item import TextParserPipelineItem


class BaseTextParser(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, list))
        self.__pipeline = pipeline

    def parse(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        for item in filter(lambda itm: itm is not None, self.__pipeline):
            assert(isinstance(item, TextParserPipelineItem))
            item.apply(pipeline_ctx)

        return BaseParsedText(terms=pipeline_ctx.provide("src"))
