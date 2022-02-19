from arekit.common.pipeline.items.base import BasePipelineItem


class FlattenIterPipelineItem(BasePipelineItem):
    """ Considered to flat iterations of items that represent iterations.
    """

    def __flat_iter(self, iter_data):
        for iter_item in iter_data:
            for item in iter_item:
                yield item

    def apply_core(self, input_data, pipeline_ctx):
        return self.__flat_iter(input_data)