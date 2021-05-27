class BaseModelIO(object):
    """
    Provides a base API for input/output operations in model
    """

    def get_model_name(self):
        raise NotImplementedError()
