from arekit.common.text.partitioning.base import BasePartitioning
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class SentenceObjectsParserPipelineItem(BasePipelineItem):

    def __init__(self, partitioning):
        assert(isinstance(partitioning, BasePartitioning))
        self.__partitioning = partitioning

    # region protected

    def _get_text(self, pipeline_ctx):
        raise NotImplementedError()

    def _get_parts_provider_func(self, pipeline_ctx):
        raise NotImplementedError()

    # endregion

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        parts = self.__partitioning.provide(
            text=self._get_text(pipeline_ctx),
            parts_it=self._get_parts_provider_func(pipeline_ctx))

        # update information in pipeline
        pipeline_ctx.update("src", parts)

    # region base

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # endregion
