from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class BasePipeline(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, list))
        self.__pipeline = pipeline

    def run(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        for item in filter(lambda itm: itm is not None, self.__pipeline):
            assert(isinstance(item, BasePipelineItem))
            item.apply(pipeline_ctx)
