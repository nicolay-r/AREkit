from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.partitioning.base import BasePartitioning


class SentenceObjectsParserPipelineItem(BasePipelineItem):

    def __init__(self, partitioning, **kwargs):
        assert(isinstance(partitioning, BasePartitioning))
        super(SentenceObjectsParserPipelineItem, self).__init__(**kwargs)
        self.__partitioning = partitioning

    # region protected

    def _get_text(self, sentence):
        return None

    def _get_parts_provider_func(self, sentence):
        raise NotImplementedError()

    # endregion

    def apply_core(self, input_data, pipeline_ctx):
        external_input = self._get_text(input_data)
        actual_input = input_data if external_input is None else external_input
        parts_it = self._get_parts_provider_func(input_data)
        return self.__partitioning.provide(text=actual_input, parts_it=parts_it)

    # region base

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # endregion
