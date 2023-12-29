from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class BasePipeline(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, list))
        self._pipeline = pipeline

    def run(self, pipeline_ctx, src_key=None):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(isinstance(src_key, str) or src_key is None)

        for ind, item in enumerate(filter(lambda itm: itm is not None, self._pipeline)):
            assert(isinstance(item, BasePipelineItem))
            input_data = item.get_source(pipeline_ctx, force_key=src_key if src_key is not None and ind == 0 else None)
            item_result = item.apply(input_data=input_data, pipeline_ctx=pipeline_ctx)
            pipeline_ctx.update(param=item.ResultKey, value=item_result, is_new_key=False)

        return pipeline_ctx

    def append(self, item):
        assert(isinstance(item, BasePipelineItem))
        self._pipeline.append(item)
