from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class BatchingPipeline(BasePipeline):

    def run(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        for item in filter(lambda itm: itm is not None, self._pipeline):
            assert (isinstance(item, BasePipelineItem))

            # Handle the content of the batch or batch itself.
            if item.SupportBatching:
                handled_batch = item.get_source(pipeline_ctx)
            else:
                content = item.get_source(pipeline_ctx, call_func=False)
                handled_batch = [item._src_func(i) if item._src_func is not None else i for i in content]

            # At present, each batch represent a list of contents.
            assert(isinstance(handled_batch, list))

            batch_result = []
            input_data_iter = [handled_batch] if item.SupportBatching else handled_batch
            for input_data in input_data_iter:
                item_result = item.apply(input_data=input_data, pipeline_ctx=pipeline_ctx)
                batch_result.append(item_result)

            pipeline_ctx.update(param=item.ResultKey, value=batch_result, is_new_key=False)

        return pipeline_ctx
