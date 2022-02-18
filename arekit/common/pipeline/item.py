from arekit.common.pipeline.context import PipelineContext


class BasePipelineItem(object):
    """ Single pipeline item that might be instatiated and embedded into pipeline.
    """

    SOURCE_KEY = "src"

    def apply_core(self, input_data, pipeline_ctx):
        raise NotImplementedError()

    def apply(self, pipeline_ctx):
        """ Performs input processing an update it for a further pipeline items.
        """
        assert(isinstance(pipeline_ctx, PipelineContext))

        result_data = self.apply_core(input_data=pipeline_ctx.provide(self.SOURCE_KEY),
                                      pipeline_ctx=pipeline_ctx)

        # Perform updating in case when we received non None result.
        if result_data is not None:
            pipeline_ctx.update(param=self.SOURCE_KEY,
                                value=result_data)
