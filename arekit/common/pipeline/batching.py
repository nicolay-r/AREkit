from arekit.common.pipeline.context import PipelineContext


class BatchingPipelineLauncher:

    @staticmethod
    def run(pipeline, pipeline_ctx, src_key=None):
        assert(isinstance(pipeline, list))
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(isinstance(src_key, str) or src_key is None)

        for ind, item in enumerate(filter(lambda itm: itm is not None, pipeline)):
            # Handle the content of the batch or batch itself.
            content = item.get_source(pipeline_ctx, call_func=False, force_key=src_key if ind == 0 else None)
            handled_batch = [item._src_func(i) if item._src_func is not None else i for i in content]

            if item.SupportBatching:
                batch_result = list(item.apply(input_data=handled_batch, pipeline_ctx=pipeline_ctx))
            else:
                batch_result = [item.apply(input_data=input_data, pipeline_ctx=pipeline_ctx)
                                for input_data in handled_batch]

            pipeline_ctx.update(param=item.ResultKey, value=batch_result, is_new_key=False)

        return pipeline_ctx