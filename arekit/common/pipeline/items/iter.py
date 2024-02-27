from arekit.common.pipeline.items.base import BasePipelineItem


class FilterPipelineItem(BasePipelineItem):

    def __init__(self, filter_func=None, **kwargs):
        assert(callable(filter_func))
        super(FilterPipelineItem, self).__init__(**kwargs)
        self.__filter_func = filter_func

    def apply_core(self, input_data, pipeline_ctx):
        return filter(self.__filter_func, input_data)