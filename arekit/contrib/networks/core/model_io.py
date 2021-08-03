from os.path import join

from arekit.common.model.model_io import BaseModelIO


class NeuralNetworkModelIO(BaseModelIO):
    """ Provides an API for saving model states
    """

    def __init__(self, target_dir, full_model_name, model_name_tag,
                 source_dir=None,
                 embedding_filepath=None,
                 vocab_filepath=None):
        assert(isinstance(target_dir, unicode))
        assert(isinstance(full_model_name, unicode))
        assert(isinstance(model_name_tag, unicode))
        assert(isinstance(source_dir, unicode) or source_dir is None)

        self.__target_dir = target_dir
        self.__full_model_name = full_model_name
        self.__model_name_tag = model_name_tag

        # States related parameters that allows to load an existed
        # model and provide all the related information for further
        # fine-tuning operation.
        self.__source_dir = source_dir
        self.__embedding_filepath = embedding_filepath
        self.__vocab_filepath = vocab_filepath

    @property
    def IsPretrainedStateProvided(self):
        return self.__is_pretrained_state_provided()

    # region private methods

    def __compose_suffixed_full_model_name(self):
        # We separate models that were trained from scratch
        # from those that adopt a pre-trained state;
        # we provide a suffix '-ft' in case of the latter.
        suffix = u"-ft" if self.__is_pretrained_state_provided() else u''
        model_tag = u"-{}".format(self.__model_name_tag) if len(self.__model_name_tag) > 0 else u''
        return self.__full_model_name + suffix + model_tag

    def __get_target_subdir(self):
        return join(self.__target_dir, self.__compose_suffixed_full_model_name())

    def __is_pretrained_state_provided(self):
        return self.__source_dir is not None

    # endregion

    def get_model_name(self):
        return self.__compose_suffixed_full_model_name()

    def get_model_dir(self):
        return self.__get_target_subdir()

    def get_model_embedding_filepath(self):
        return self.__embedding_filepath

    def get_model_vocab_filepath(self):
        return self.__vocab_filepath

    def get_model_source_path_tf_prefix(self):
        """ Provides a filepath to the state that should be utilized as original.
        """
        return join(self.__source_dir, self.__full_model_name)

    def get_model_target_path_tf_prefix(self):
        """ Provides the template for states keeping
            using tensorflow serializer.
        """
        return join(self.__get_target_subdir(), self.__full_model_name)
