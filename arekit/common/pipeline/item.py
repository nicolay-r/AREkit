from arekit.common.pipeline.context import PipelineContext


class BasePipelineItem(object):
    """ Single pipeline item that might be instatiated and embedded into pipeline.
    """

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        raise NotImplementedError()
