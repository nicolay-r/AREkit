from os.path import join

from arekit.common.model.model_io import BaseModelIO


class NeuralNetworkModelIO(BaseModelIO):
    """ Provides an API for saving model states
    """

    def __init__(self, model_dir, full_model_name, load_dir):
        assert(isinstance(model_dir, unicode))
        assert(isinstance(full_model_name, unicode))
        assert(isinstance(load_dir, unicode) or load_dir is None)
        self.__model_dir = model_dir
        self.__full_model_name = full_model_name
        self.__load_dir = load_dir

    # region private methods

    def __get_model_dir(self):
        return join(self.__model_dir, self.__full_model_name)

    # endregion

    def get_model_dir(self):
        return self.__get_model_dir()

    def get_model_load_path_tf_prefix(self):
        """ Provides a filepath to the state that should be
            utilized an an original.
        """
        return join(self.__load_dir, self.__full_model_name)

    def get_model_save_path_tf_prefix(self):
        """ Provides the template for states keeping
            using tensorflow serializer.
        """
        return join(self.__get_model_dir(), self.__full_model_name)
