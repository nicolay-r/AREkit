from arekit.common.pipeline.items.map import MapPipelineItem


class MapNestedPipelineItem(MapPipelineItem):
    """ This type is considered for describing nested pipelines,
        which might be required in parameters of the parent pipeline-contexts.

        Data treated as a sequence, in which every element is
        suppose to be mapped with the passed pipeline context.
    """

    def apply_core(self, input_data, pipeline_ctx):
        return map(lambda item: self._map_func(item, pipeline_ctx), input_data)
