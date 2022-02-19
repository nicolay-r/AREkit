from arekit.common.pipeline.items.base import BasePipelineItem


class HandleIterPipelineItem(BasePipelineItem):

    def __init__(self, handle_func=None):
        assert(callable(handle_func))
        self.__handle_func = handle_func

    def __updated_data(self, items_iter):
        for item in items_iter:
            # Perform item handling
            self.__handle_func(item)
            yield item

    def apply_core(self, input_data, pipeline_ctx):
        return self.__updated_data(input_data)