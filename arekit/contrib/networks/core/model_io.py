from os import listdir
from os.path import join, exists


class TensorflowNeuralNetworkModelIO(object):
    """ Provides an API for saving model states
    """

    def __init__(self, model_name, target_dir=None, source_dir=None):
        """ model_name_tag: str
                general model suffix to be adopted if it is specified.
                allows to customize your model name.
            finetuned_suffix: str
                utilized to specify that the new model represents a finetuned version of
                the previously existed.
        """
        assert(isinstance(model_name, str))
        assert(isinstance(target_dir, str) or target_dir is None)
        assert(isinstance(source_dir, str) or source_dir is None)

        if isinstance(source_dir, str) and isinstance(target_dir, str):
            if source_dir == target_dir:
                raise Exception("source_dir and taget_dir could not be "
                                "exact the same value! (Not supported)")

        self.__target_dir = target_dir
        self.__model_name = model_name

        # States related parameters that allows to load an existed
        # model and provide all the related information for further
        # fine-tuning operation.
        self.__source_dir = source_dir

    @property
    def IsPretrainedStateProvided(self):
        """ NOTE: We consider that if the folder with the original model directory
            could be found and this folder is not empty in terms of the its
            contents, then we return True; False otherwise.
        """
        if self.__source_dir is not None:
            source_model_dir = self.__get_original_state_model_dir()
            if exists(source_model_dir) and len(listdir(source_model_dir)) > 0:
                return True
        return False

    # region private methods

    def __get_original_state_model_dir(self):
        return join(self.__source_dir, self.__model_name)

    # endregion

    def get_model_saving_dir(self):
        return join(self.__target_dir, self.__model_name)

    def get_model_source_path_tf_prefix(self):
        """ Path, utiliized for the fine-tunning, i.e. original model state. (Loading)
        """
        return self.__get_original_state_model_dir()

    def get_model_target_path_tf_prefix(self):
        """ Provides the template for states keeping using tensorflow serializer. (Saving)
        """
        return join(self.__target_dir, self.__model_name, self.__model_name)
