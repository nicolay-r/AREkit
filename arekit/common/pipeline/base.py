from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class BasePipelineLauncher:

    @staticmethod
    def run(pipeline, pipeline_ctx, src_key=None):
        assert(isinstance(pipeline, list))
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(isinstance(src_key, str) or src_key is None)

        for ind, item in enumerate(filter(lambda itm: itm is not None, pipeline)):
            assert(isinstance(item, BasePipelineItem))
            input_data = item.get_source(pipeline_ctx, force_key=src_key if src_key is not None and ind == 0 else None)
            item_result = item.apply(input_data=input_data, pipeline_ctx=pipeline_ctx)
            pipeline_ctx.update(param=item.ResultKey, value=item_result, is_new_key=False)

        return pipeline_ctx
