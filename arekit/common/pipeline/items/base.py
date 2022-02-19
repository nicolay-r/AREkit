class BasePipelineItem(object):
    """ Single pipeline item that might be instatiated and embedded into pipeline.
    """

    def apply_core(self, input_data, pipeline_ctx):
        raise NotImplementedError()

    def apply(self, input_data, pipeline_ctx=None):
        """ Performs input processing an update it for a further pipeline items.
        """
        output_data = self.apply_core(input_data=input_data, pipeline_ctx=pipeline_ctx)
        return output_data if output_data is not None else input_data
