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
from arekit.contrib.experiment_rusentrel.model_io.utils import join_dir_with_subfolder_name
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.np_utils.embedding import EmbeddingHelper
from arekit.contrib.source.rusentrel.opinions.provider import RuSentRelOpinionCollectionProvider
from arekit.contrib.source.rusentrel.opinions.writer import RuSentRelOpinionCollectionWriter


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelExperimentNetworkIOUtils(BaseIOUtils):
    """ Provides additional Input/Output paths generation functions for:
        - model directory;
        - embedding matrix;
        - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = 'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = "vocab-{cv_index}.txt"

    # region public methods

    def create_docs_stat_target(self):
        return join(self._get_target_dir(), "docs_stat.txt")

    def get_experiment_folder_name(self):
        return self.__get_experiment_folder_name()

    def get_output_model_results_filepath(self, data_type, epoch_index):
        f_name_template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.__get_model_dir(),
                                  template="".join([f_name_template, '-e{e_index}'.format(e_index=epoch_index)]),
                                  prefix="result")

    def create_samples_view(self, data_type):
        assert(isinstance(data_type, DataType))
        storage = BaseRowsStorage.from_tsv(
            filepath=self.__get_input_sample_target(data_type=data_type))

        return BaseSampleStorageView(storage=storage,
                                     row_ids_provider=MultipleIDProvider())

    def create_opinions_view(self, data_type):
        assert(isinstance(data_type, DataType))
        storage = BaseRowsStorage.from_tsv(
            filepath=self.__get_input_opinions_target(data_type=data_type))
        return BaseOpinionStorageView(storage)

    def create_opinions_writer(self):
        return TsvWriter(write_header=False)

    def create_samples_writer(self):
        return TsvWriter(write_header=True)

    def create_opinions_writer_target(self, data_type):
        return self.__get_input_opinions_target(data_type)

    def create_samples_writer_target(self, data_type):
        return self.__get_input_sample_target(data_type)

    def save_vocab(self, data):
        target = self.__get_default_vocab_filepath()
        return EmbeddingHelper.save_vocab(data=data, target=target)

    def load_vocab(self):
        source = self.___get_vocab_source()
        return EmbeddingHelper.load_vocab(source)

    def save_embedding(self, data):
        target = self.__get_default_embedding_filepath()
        EmbeddingHelper.save_embedding(data=data, target=target)

    def load_embedding(self):
        source = self.__get_term_embedding_source()
        return EmbeddingHelper.load_embedding(source)

    def has_model_predefined_state(self):
        model_io = self._experiment.DataIO.ModelIO
        return self.__model_is_pretrained_state_provided(model_io)

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        assert(isinstance(epoch_index, int))
        model_eval_root = self.__get_eval_root_filepath(data_type=data_type, epoch_index=epoch_index)
        return join(model_eval_root, "{}.opin.txt".format(doc_id))

    def check_targets_existed(self, data_types_iter):
        for data_type in data_types_iter:

            filepaths = [
                self.__get_input_sample_target(data_type=data_type),
                self.__get_input_opinions_target(data_type=data_type),
                self.__get_default_vocab_filepath(),
                self.__get_term_embedding_target()
            ]

            if not self.__check_targets_existence(targets=filepaths, logger=logger):
                return False
        return True

    # endregion

    # region private methods

    def __get_input_opinions_target(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self._get_target_dir(), template=template, prefix="opinion")

    def __get_input_sample_target(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self._get_target_dir(), template=template, prefix="sample")

    def __get_term_embedding_target(self):
        return self.__get_default_embedding_filepath()

    @staticmethod
    def __model_is_pretrained_state_provided(model_io):
        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.IsPretrainedStateProvided

    def ___get_vocab_source(self):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        model_io = self._experiment.DataIO.ModelIO

        if model_io is None:
            return self.__get_default_vocab_filepath()

        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.get_model_vocab_filepath() if self.__model_is_pretrained_state_provided(model_io) \
            else self.__get_default_vocab_filepath()

    def __get_term_embedding_source(self):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        model_io = self._experiment.DataIO.ModelIO

        if model_io is None:
            return self.__get_default_embedding_filepath()

        assert(isinstance(model_io, NeuralNetworkModelIO))
        return model_io.get_model_embedding_filepath() if self.__model_is_pretrained_state_provided(model_io) \
            else self.__get_default_embedding_filepath()

    def __get_experiment_folder_name(self):
        return "{name}_{scale}l".format(name=self._experiment.DataIO.Name,
                                        scale=str(self._experiment.DataIO.LabelsCount))

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

    def __get_default_vocab_filepath(self):
        return join(self._get_target_dir(),
                    self.VOCABULARY_FILENAME_TEMPLATE.format(
                        cv_index=self._experiment_iter_index()) + '.npz')

    def __get_default_embedding_filepath(self):
        return join(self._get_target_dir(),
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

    def __get_annotator_dir(self):
        return join_dir_with_subfolder_name(dir=self._get_target_dir(),
                                            subfolder_name=self._get_annotator_name())

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

    def _create_opinion_collection_provider(self):
        return RuSentRelOpinionCollectionProvider()

    def _create_opinion_collection_writer(self):
        return RuSentRelOpinionCollectionWriter()

    @staticmethod
    def _get_filepath(out_dir, template, prefix):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        return join(out_dir, RuSentRelExperimentNetworkIOUtils.__generate_tsv_archive_filename(template=template, prefix=prefix))

    def _experiment_iter_index(self):
        return self._experiment.DocumentOperations.DataFolding.IterationIndex

    def _filename_template(self, data_type):
        assert(isinstance(data_type, DataType))
        return "{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                                 iter_index=self._experiment_iter_index())

    def _get_annotator_name(self):
        """ We use custom implementation as it allows to
            be independent of NeutralAnnotator instance.
        """
        return "annot_{labels_count}l".format(labels_count=self._experiment.DataIO.LabelsCount)

    def _create_annotated_collection_target(self, doc_id, data_type, check_existance):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))
        assert(isinstance(check_existance, bool))

        annot_dir = self.__get_annotator_dir()

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        # TODO. This should not depends on the neut.
        filename = "art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                          d_type=data_type.name)

        target = join(annot_dir, filename)

        if check_existance and not exists(target):
            return None

        return target

    # endregion
