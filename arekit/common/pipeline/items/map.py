from arekit.common.pipeline.items.base import BasePipelineItem


class MapPipelineItem(BasePipelineItem):

    def __init__(self, map_func=None):
        assert(callable(map_func))
        self._map_func = map_func

    def apply_core(self, input_data, pipeline_ctx):
        return map(self._map_func, input_data)
