import logging
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.tsv_opinion import TsvInputOpinionReader
from arekit.common.experiment.input.readers.tsv_sample import TsvInputSampleReader
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkIOUtils(BaseIOUtils):
    """ Provides additional Input/Output paths generation functions for:
        - model directory;
        - embedding matrix;
        - embedding vocabulary.
    """

    # TODO. Move it outside.
    # TODO. Move it outside.
    # TODO. Move it outside.
    TERM_EMBEDDING_FILENAME_TEMPLATE = 'term_embedding-{cv_index}'
    # TODO. Move it outside too.
    # TODO. Move it outside too.
    # TODO. Move it outside too.
    VOCABULARY_FILENAME_TEMPLATE = "vocab-{cv_index}.txt"

    # region public methods

    def create_samples_reader(self, data_type):
        assert(isinstance(data_type, DataType))

        return TsvInputSampleReader.from_tsv(
            filepath=self.get_input_sample_filepath(data_type=data_type),
            row_ids_provider=MultipleIDProvider())

    def create_opinions_reader(self, data_type):
        assert(isinstance(data_type, DataType))

        opinions_source = self.get_input_opinions_filepath(data_type=data_type)
        return TsvInputOpinionReader.from_tsv(opinions_source)

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

    # TODO. Filepath-dependency should be removed!
    # TODO. Filepath-dependency should be removed!
    # TODO. Filepath-dependency should be removed!
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

    # TODO. In nested class (user applications)
    def get_input_opinions_filepath(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.get_target_dir(),
                                  template=template,
                                  # TODO. formatter_type_log_name -- in nested formatter.
                                  prefix="opinion")

    # TODO. In nested class (user applications)
    def get_input_sample_filepath(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.get_target_dir(),
                                  template=template,
                                  # TODO. formatter_type_log_name -- in nested formatter.
                                  prefix="sample")

    # TODO. In nested class (user applications)
    @staticmethod
    def _get_filepath(out_dir, template, prefix):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        return join(out_dir, NetworkIOUtils.__generate_tsv_archive_filename(template=template, prefix=prefix))

    # TODO. In nested class (user applications)
    def _experiment_iter_index(self):
        return self._experiment.DocumentOperations.DataFolding.IterationIndex

    # TODO. In nested class (user applications)
    def _filename_template(self, data_type):
        assert(isinstance(data_type, DataType))
        return "{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                                 iter_index=self._experiment_iter_index())

    # TODO. In nested class (user applications)
    @staticmethod
    def __generate_tsv_archive_filename(template, prefix):
        return "{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

