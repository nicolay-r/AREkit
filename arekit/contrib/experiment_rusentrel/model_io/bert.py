import logging
from os.path import join, exists

from arekit.common.data import const
from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.opinions import BaseOpinionStorageView
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.contrib.bert.output.eval_helper import EvalHelper
from arekit.contrib.bert.output.google_bert_provider import GoogleBertOutputStorage
from arekit.contrib.experiment_rusentrel.model_io.utils import join_dir_with_subfolder_name, experiment_iter_index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelExperimentBertIOUtils(BaseIOUtils):

    def _get_experiment_sources_dir(self):
        """ Provides directory for samples.
        """
        raise NotImplementedError()

    def create_docs_stat_target(self):
        return join(self.__get_target_dir(), "docs_stat.txt")

    def get_output_storage(self, epoch_index, iter_index, eval_helper):
        assert(isinstance(eval_helper, EvalHelper))

        # NOTE: we wrap original dir using eval_helper implementation.
        # The latter allows us support a custom dir modifications while all the
        # required data stays unchanged in terms of paths.
        original_target_dir = self.__get_target_dir()
        target_dir = eval_helper.get_results_dir(original_target_dir)

        result_filename = eval_helper.get_results_target(
            iter_index=iter_index,
            epoch_index=epoch_index)

        result_filepath = join(target_dir, result_filename)

        if not exists(result_filepath):
            logger.info("Result filepath was not found: {}".format(result_filepath))
            return None

        # Initialize storage.
        output_storage = GoogleBertOutputStorage.from_tsv(filepath=result_filepath, header=None)
        output_storage.apply_samples_view(
            row_ids=output_storage.iter_column_values(column_name=const.ID, dtype=str),
            doc_ids=output_storage.iter_column_values(column_name=const.DOC_ID, dtype=str))

        return output_storage

    def try_prepare(self):
        model_dir = self.__get_target_dir()
        if not exists(model_dir):
            logger.info("Model dir does not exist. Skipping")
            return False

        exp_dir = join_dir_with_subfolder_name(
            subfolder_name=self.__get_experiment_folder_name(),
            dir=self._get_experiment_sources_dir())

        if not exists(exp_dir):
            logger.info("Experiment dir: {}".format(exp_dir))
            logger.info("Experiment dir does not exist. Skipping")
            return False

        return

    # region experiment dir related

    def __get_target_dir(self):
        """ Provides a main directory for input

            NOTE:
            We consider to save serialized results into model dir,
            rather than experiment dir in a base implementation,
            as model affects on text_b, entities representation, etc.
        """
        default_dir = join_dir_with_subfolder_name(
            subfolder_name=self.__get_experiment_folder_name(),
            dir=self._get_experiment_sources_dir())

        return join(default_dir, self._exp_ctx.ModelIO.get_model_name())

    def __get_experiment_folder_name(self):
        return "{name}_{scale}l".format(name=self._exp_ctx.Name,
                                        scale=str(self._exp_ctx.LabelsCount))

    # endregion

    # region public methods

    def create_samples_view(self, data_type):
        return BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv(filepath=self.__get_input_sample_filepath(data_type)),
            row_ids_provider=MultipleIDProvider())

    def create_opinions_view(self, data_type):
        storage = BaseRowsStorage.from_tsv(
            filepath=self.__get_input_opinions_filepath(data_type),
            compression='infer')

        return BaseOpinionStorageView(storage=storage)

    def create_opinions_writer_target(self, data_type):
        return self.__get_input_opinions_filepath(data_type)

    def create_samples_writer_target(self, data_type):
        return self.__get_input_sample_filepath(data_type)

    def create_samples_writer(self):
        return TsvWriter(write_header=True)

    def create_opinions_writer(self):
        return TsvWriter(write_header=False)

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        """ Utilized for results evaluation.
        """
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type, epoch_index=epoch_index)

        filepath = join(model_eval_root, "{}.opin.txt".format(doc_id))

        return filepath

    # endregion

    # region private methods (filepaths)

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        result_dir = join(
            self.__get_target_dir(),
            join("eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type.name,
                iter_index=experiment_iter_index(self._exp_ctx.DataFolding),
                epoch_index=str(epoch_index))))

        return result_dir

    def __get_input_opinions_filepath(self, data_type):
        template = self.__filename_template(data_type=data_type)
        return self.__get_filepath(out_dir=self.__get_target_dir(), template=template, prefix="opinion")

    def __get_input_sample_filepath(self, data_type):
        template = self.__filename_template(data_type=data_type)
        return self.__get_filepath(out_dir=self.__get_target_dir(), template=template, prefix="sample")

    @staticmethod
    def __get_filepath(out_dir, template, prefix):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        return join(out_dir, RuSentRelExperimentBertIOUtils.__generate_tsv_archive_filename(template=template, prefix=prefix))

    @staticmethod
    def __generate_tsv_archive_filename(template, prefix):
        return "{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

    def __get_annotator_dir(self):
        return join_dir_with_subfolder_name(dir=self.__get_target_dir(),
                                            subfolder_name=self.__get_annotator_name())

    def __filename_template(self, data_type):
        assert(isinstance(data_type, DataType))
        return "{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                                 iter_index=experiment_iter_index(self._exp_ctx.DataFolding))

    def __get_annotator_name(self):
        """ We use custom implementation as it allows to
            be independent of NeutralAnnotator instance.
        """
        return "annot_{labels_count}l".format(labels_count=self._exp_ctx.LabelsCount)

    # endregion

    # region protected methods

    def _create_annotated_collection_target(self, doc_id, data_type, check_existence):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))
        assert(isinstance(check_existence, bool))

        annot_dir = self.__get_annotator_dir()

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        # TODO. This should not depends on the neut.
        filename = "art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                          d_type=data_type.name)

        target = join(annot_dir, filename)

        if check_existence and not exists(target):
            return None

        return target

    # endregion
