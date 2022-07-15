import collections
import logging
from os.path import join, exists

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.opinions import BaseOpinionStorageView
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.utils.model_io.utils import join_dir_with_subfolder_name, filename_template
from arekit.contrib.utils.utils_folding import experiment_iter_index
from arekit.contrib.utils.np_utils.embedding import NpzEmbeddingHelper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DefaultNetworkIOUtils(BaseIOUtils):
    """ This is a default file-based Input-output utils,
        which describes file-paths towards the resources, required
        for BERT-related data preparation.

        Provides additional Input/Output paths generation functions for:
            - model directory;
            - embedding matrix;
            - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = 'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = "vocab-{cv_index}.txt"

    # region public methods

    def get_target_dir(self):
        return self._get_target_dir()

    def get_experiment_folder_name(self):
        return self.__get_experiment_folder_name()

    def create_samples_view(self, data_type, data_folding):
        assert(isinstance(data_type, DataType))
        storage = BaseRowsStorage.from_tsv(
            filepath=self.__get_input_sample_target(data_type=data_type, data_folding=data_folding))
        return BaseSampleStorageView(storage=storage,
                                     row_ids_provider=MultipleIDProvider())

    def create_opinions_view(self, target):
        storage = BaseRowsStorage.from_tsv(filepath=target)
        return BaseOpinionStorageView(storage)

    def create_opinions_writer(self):
        return TsvWriter(write_header=False)

    def create_samples_writer(self):
        return TsvWriter(write_header=True)

    def create_opinions_writer_target(self, data_type, data_folding):
        return self.__get_input_opinions_target(data_type, data_folding=data_folding)

    def create_samples_writer_target(self, data_type, data_folding):
        return self.__get_input_sample_target(data_type, data_folding=data_folding)

    def save_vocab(self, data, data_folding):
        assert(isinstance(data_folding, BaseDataFolding))
        target = self.__get_default_vocab_filepath(data_folding)
        return NpzEmbeddingHelper.save_vocab(data=data, target=target)

    def load_vocab(self, data_folding):
        source = self.___get_vocab_source(data_folding)
        return NpzEmbeddingHelper.load_vocab(source)

    def save_embedding(self, data, data_folding):
        assert(isinstance(data_folding, BaseDataFolding))
        target = self.__get_default_embedding_filepath(data_folding)
        NpzEmbeddingHelper.save_embedding(data=data, target=target)

    def load_embedding(self, data_folding):
        source = self.__get_term_embedding_source(data_folding)
        return NpzEmbeddingHelper.load_embedding(source)

    def has_model_predefined_state(self):
        model_io = self._exp_ctx.ModelIO
        return self.__model_is_pretrained_state_provided(model_io)

    def check_targets_existed(self, data_types_iter, data_folding):
        for data_type in data_types_iter:

            filepaths = [
                self.__get_input_sample_target(data_type=data_type, data_folding=data_folding),
                self.__get_default_vocab_filepath(data_folding=data_folding),
                self.__get_term_embedding_target(data_folding=data_folding)
            ]

            if not self.__check_targets_existence(targets=filepaths, logger=logger):
                return False
        return True

    # endregion

    # region private methods

    def __get_model_parameter(self, default_value, get_value_func):
        assert(default_value is not None)
        assert(callable(get_value_func))

        model_io = self._exp_ctx.ModelIO

        if model_io is None:
            return default_value

        predefined_value = get_value_func(model_io) if \
            self.__model_is_pretrained_state_provided(model_io) else None

        return default_value if predefined_value is None else predefined_value

    def __get_input_opinions_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self._get_filepath(out_dir=self._get_target_dir(), template=template, prefix="opinion")

    def __get_input_sample_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self._get_filepath(out_dir=self._get_target_dir(), template=template, prefix="sample")

    def __get_term_embedding_target(self, data_folding):
        return self.__get_default_embedding_filepath(data_folding)

    @staticmethod
    def __model_is_pretrained_state_provided(model_io):
        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.IsPretrainedStateProvided

    def ___get_vocab_source(self, data_folding):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        return self.__get_model_parameter(default_value=self.__get_default_vocab_filepath(data_folding),
                                          get_value_func=lambda model_io: model_io.get_model_vocab_filepath())

    def __get_term_embedding_source(self, data_folding):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        return self.__get_model_parameter(default_value=self.__get_default_embedding_filepath(data_folding),
                                          get_value_func=lambda model_io: model_io.get_model_embedding_filepath())

    def __get_experiment_folder_name(self):
        return "{name}_{scale}l".format(name=self._exp_ctx.Name,
                                        scale=str(self._exp_ctx.LabelsCount))

    @staticmethod
    def __generate_tsv_archive_filename(template, prefix):
        return "{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

    @staticmethod
    def __check_targets_existence(targets, logger):
        assert (isinstance(targets, collections.Iterable))

        result = True
        for filepath in targets:
            assert(isinstance(filepath, str))

            existed = exists(filepath)
            logger.info("Check existance [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
            if not existed:
                result = False

        return result

    def __get_default_vocab_filepath(self, data_folding):
        return join(self._get_target_dir(),
                    self.VOCABULARY_FILENAME_TEMPLATE.format(
                        cv_index=experiment_iter_index(data_folding)) + '.npz')

    def __get_default_embedding_filepath(self, data_folding):
        return join(self._get_target_dir(),
                    self.TERM_EMBEDDING_FILENAME_TEMPLATE.format(
                        cv_index=experiment_iter_index(data_folding)) + '.npz')

    def __get_model_dir(self):
        # Perform access to the model, since all the IO information
        # that is related to the model, assumes to be stored in ModelIO.
        model_io = self._exp_ctx.ModelIO
        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.get_model_dir()

    # endregion

    # region protected methods

    def _get_experiment_sources_dir(self):
        raise NotImplementedError()

    def _get_target_dir(self):
        """ Represents an experiment dir of specific label scale format,
            defined by labels scaler.
        """
        return join_dir_with_subfolder_name(subfolder_name=self.__get_experiment_folder_name(),
                                            dir=self._get_experiment_sources_dir())

    @staticmethod
    def _get_filepath(out_dir, template, prefix):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        return join(out_dir, DefaultNetworkIOUtils.__generate_tsv_archive_filename(template=template, prefix=prefix))

    # endregion
