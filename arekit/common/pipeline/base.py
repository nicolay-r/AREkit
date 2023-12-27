from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class BasePipeline(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, list))
        self.__pipeline = pipeline

    def run(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        for item in filter(lambda itm: itm is not None, self.__pipeline):
            assert(isinstance(item, BasePipelineItem))
            item_result = item.apply(input_data=item.get_source(pipeline_ctx), pipeline_ctx=pipeline_ctx)
            pipeline_ctx.update(param=item.ResultKey, value=item_result, is_new_key=False)

        return pipeline_ctx

    def append(self, item):
        assert(isinstance(item, BasePipelineItem))
        self.__pipeline.append(item)
