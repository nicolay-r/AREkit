from os.path import join

from arekit.common.model.model_io import BaseModelIO


class NeuralNetworkModelIO(BaseModelIO):
    """ Provides an API for saving model states
    """

    def __init__(self, target_dir, full_model_name, source_dir):
        assert(isinstance(target_dir, unicode))
        assert(isinstance(full_model_name, unicode))
        assert(isinstance(source_dir, unicode) or source_dir is None)
        self.__target_dir = target_dir
        self.__full_model_name = full_model_name
        self.__source_dir = source_dir

    @property
    def IsPretrainedStateProvided(self):
        return self.__is_pretrained_state_provided()

    # region private methods

    def __get_target_subdir(self):
        # We separate models that were trained from scratch
        # from those that adopt a pre-trained state;
        # we provide suffix '-tf' in case of the latter.
        suffix = u"-ft" if self.__is_pretrained_state_provided() else ""
        return join(self.__target_dir, self.__full_model_name + suffix)

    def __is_pretrained_state_provided(self):
        return self.__source_dir is not None

    # endregion

    def get_model_name(self):
        return self.__full_model_name

    def get_model_dir(self):
        return self.__get_target_subdir()

    def get_model_source_path_tf_prefix(self):
        """ Provides a filepath to the state that should be
            utilized an an original.
        """
        return join(self.__source_dir, self.__full_model_name)

    def get_model_target_path_tf_prefix(self):
        """ Provides the template for states keeping
            using tensorflow serializer.
        """
        return join(self.__get_target_subdir(), self.__full_model_name)
