from os.path import join

from arekit.common.model.model_io import BaseModelIO


class NeuralNetworkModelIO(BaseModelIO):
    """ Provides an API for saving model states
    """

    def __init__(self, model_dir, full_model_name):
        self.__model_dir = model_dir
        self.__full_model_name = full_model_name

    def get_model_dir(self):
        return join(self.__model_dir, self.__full_model_name)

    def get_model_save_path_tf_prefix(self):
        """ Provides the template for states keeping
            using tensorflow serializer.
        """
        return join(self.__model_dir, self.__full_model_name)
