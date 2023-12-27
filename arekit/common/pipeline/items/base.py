from arekit.common.pipeline.context import PipelineContext


class BasePipelineItem(object):
    """ Single pipeline item that might be instatiated and embedded into pipeline.
    """

    def __init__(self, src_key="result", result_key="result", src_func=None):
        assert(isinstance(src_key, str) or src_key is None)
        assert(callable(src_func) or src_func is None)
        self.__src_key = src_key
        self.__src_func = src_func
        self.__result_key = result_key

    @property
    def ResultKey(self):
        return self.__result_key

    def get_source(self, src_ctx):
        """ Extract input element for processing.
        """
        assert(isinstance(src_ctx, PipelineContext))

        # If there is no information about key, then we consider absence of the source.
        if self.__src_key is None:
            return None

        # Extracting actual source.
        src_data = src_ctx.provide(self.__src_key)
        if self.__src_func is not None:
            src_data = self.__src_func(src_data)

        return src_data

    def apply_core(self, input_data, pipeline_ctx):
        raise NotImplementedError()

    def apply(self, input_data, pipeline_ctx=None):
        """ Performs input processing an update it for a further pipeline items.
        """
        output_data = self.apply_core(input_data=input_data, pipeline_ctx=pipeline_ctx)
        return output_data if output_data is not None else input_data
