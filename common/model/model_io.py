class BaseModelIO(object):
    """
    Provides a base API for input/output operations in model
    """

    @property
    def ModelRoot(self):
        raise NotImplementedError()