class BaseModelIO(object):
    """
    Provides a base API for input/output operations in model
    """

    def get_model_name(self):
        raise NotImplementedError()

    # TODO. This is a part of the neural network model.
    # Therefore the latter should be moved in a nested class.
    def get_model_dir(self):
        raise NotImplementedError()
