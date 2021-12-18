class PipelineContext(object):
    """ Context of parameters utilized in pipeline
    """

    def __init__(self, d):
        assert(isinstance(d, dict))
        self._d = d

    def provide(self, param):
        return self._d[param]

    def update(self, param, value):
        self._d[param] = value

    def __contains__(self, item):
        return item in self._d