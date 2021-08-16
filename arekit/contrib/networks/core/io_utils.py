import logging
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.common.model.model_io import BaseModelIO
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkIOUtils(BaseIOUtils):
    """ Provides additional Input/Output paths generation functions for:
        - model directory;
        - embedding matrix;
        - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = 'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = "vocab-{cv_index}.txt"

    # region public methods

    def get_loading_vocab_filepath(self):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        model_io = self._experiment.DataIO.ModelIO
        assert(isinstance(model_io, NeuralNetworkModelIO))

        return model_io.get_model_vocab_filepath() if self.__model_is_pretrained_state_provided(model_io) \
            else self.__get_default_vocab_filepath()

    def get_saving_vocab_filepath(self):
        return self.__get_default_vocab_filepath()

    def has_model_predefined_state(self):
        model_io = self._experiment.DataIO.ModelIO
        return self.__model_is_pretrained_state_provided(model_io)

    def get_loading_embedding_filepath(self):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        model_io = self._experiment.DataIO.ModelIO
        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.get_model_embedding_filepath() if self.__model_is_pretrained_state_provided(model_io) \
            else self.__get_default_embedding_filepath()

    def get_saving_embedding_filepath(self):
        return self.__get_default_embedding_filepath()

    def get_output_model_results_filepath(self, data_type, epoch_index):

        f_name_template = self._filename_template(data_type=data_type)

        result_template = "".join([f_name_template, '-e{e_index}'.format(e_index=epoch_index)])

        return self._get_filepath(out_dir=self.__get_model_dir(),
                                  template=result_template,
                                  prefix="result")

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type, epoch_index=epoch_index)

        filepath = join(model_eval_root, "{}.opin.txt".format(doc_id))

        return filepath

    # endregion

    # region private methods

    def __get_default_vocab_filepath(self):
        return join(self.get_target_dir(),
                    self.VOCABULARY_FILENAME_TEMPLATE.format(
                        cv_index=self._experiment_iter_index()) + '.npz')

    def __get_default_embedding_filepath(self):
        return join(self.get_target_dir(),
                    self.TERM_EMBEDDING_FILENAME_TEMPLATE.format(
                        cv_index=self._experiment_iter_index()) + '.npz')

    def __get_model_dir(self):
        # Perform access to the model, since all the IO information
        # that is related to the model, assumes to be stored in ModelIO.
        model_io = self._experiment.DataIO.ModelIO
        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.get_model_dir()

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        result_dir = join(
            self.__get_model_dir(),
            join("eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type.name,
                iter_index=self._experiment_iter_index(),
                epoch_index=str(epoch_index))))

        return result_dir

    @staticmethod
    def __model_is_pretrained_state_provided(model_io):
        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.IsPretrainedStateProvided

    # endregion