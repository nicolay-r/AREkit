class BaseModelIO(object):
    """
    Provides a base API for input/output operations in model
    """

    @property
    def get_model_dir(self):
        raise NotImplementedError()
