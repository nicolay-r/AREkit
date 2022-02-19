from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.partitioning.base import BasePartitioning
from arekit.common.pipeline.context import PipelineContext


class SentenceObjectsParserPipelineItem(BasePipelineItem):

    def __init__(self, partitioning):
        assert(isinstance(partitioning, BasePartitioning))
        self.__partitioning = partitioning

    # region protected

    def _get_text(self, pipeline_ctx):
        return None

    def _get_parts_provider_func(self, input_data, pipeline_ctx):
        raise NotImplementedError()

    # endregion

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        external_input = self._get_text(pipeline_ctx)
        actual_input = input_data if external_input is None else external_input
        parts_it = self._get_parts_provider_func(input_data=actual_input, pipeline_ctx=pipeline_ctx)
        return self.__partitioning.provide(text=actual_input, parts_it=parts_it)

    # region base

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # endregion
