from arekit.common.pipeline.conts import PARENT_CTX


class PipelineContext(object):
    """ Context of parameters utilized in pipeline
    """

    def __init__(self, d, parent_ctx=None):
        assert(isinstance(d, dict))
        assert(isinstance(parent_ctx, PipelineContext) or parent_ctx is None)
        assert(PARENT_CTX not in d)
        self._d = d
        self._d[PARENT_CTX] = parent_ctx

    def __provide(self, param):
        if param not in self._d:
            raise Exception(f"Key `{param}` is not in dictionary.\n{self._d}")
        return self._d[param]

    # region public

    def provide(self, param):
        return self.__provide(param)

    def provide_or_none(self, param):
        return self.__provide(param) if param in self._d else None

    def update(self, param, value, is_new_key=False):
        if is_new_key and param in self._d:
            raise Exception(f"Key `{param}` is already presented in pipeline context dictionary.")
        self._d[param] = value

    # endregion

    # region base methods

    def __contains__(self, item):
        return item in self._d

    # endregion
