import logging
from os.path import join, exists

from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView
from arekit.contrib.utils.io_utils.utils import join_dir_with_subfolder_name, filename_template

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DefaultBertIOUtils(BaseIOUtils):
    """ This is a default file-based Input-output utils,
        which describes file-paths towards the resources, required
        for BERT-related data preparation.
    """

    def __init__(self, exp_ctx,
                 samples_writer=TsvWriter(write_header=True),
                 target_extension=".tsv.gz"):
        assert(isinstance(exp_ctx, ExperimentContext))
        assert(isinstance(samples_writer, BaseWriter))
        self.__exp_ctx = exp_ctx
        self.__samples_writer = samples_writer
        self.__target_extension = target_extension

    def _get_experiment_sources_dir(self):
        """ Provides directory for samples.
        """
        raise NotImplementedError()

    def check_targets_existed(self):
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

    def get_target_dir(self):
        return self.__get_target_dir()

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

        return join(default_dir, self.__exp_ctx.ModelIO.get_model_name())

    def __get_experiment_folder_name(self):
        return "{name}".format(name=self.__exp_ctx.Name)

    # endregion

    # region public methods

    def create_samples_view(self, data_type, data_folding):
        return BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv(filepath=self.__get_input_sample_filepath(
                data_type=data_type, data_folding=data_folding)),
            row_ids_provider=MultipleIDProvider())

    def create_opinions_view(self, target):
        storage = BaseRowsStorage.from_tsv(filepath=target, compression='infer')
        return BaseOpinionStorageView(storage=storage)

    def create_opinions_writer_target(self, data_type, data_folding):
        return self.__get_input_opinions_filepath(data_type, data_folding=data_folding)

    def create_samples_writer_target(self, data_type, data_folding):
        return self.__get_input_sample_filepath(data_type, data_folding=data_folding)

    def create_target_extension(self):
        return self.__target_extension

    def create_samples_writer(self):
        return self.__samples_writer

    def create_opinions_writer(self):
        return self.__samples_writer

    # endregion

    # region private methods (filepaths)

    def __get_input_opinions_filepath(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__get_target_dir(),
                                   template=template, prefix="opinion", extension=self.create_target_extension())

    def __get_input_sample_filepath(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__get_target_dir(),
                                   template=template, prefix="sample", extension=self.create_target_extension())

    @staticmethod
    def __get_filepath(out_dir, template, prefix, extension):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        assert(isinstance(extension, str))
        return join(out_dir, "{prefix}-{template}{extension}".format(
            prefix=prefix, template=template, extension=extension))

    # endregion
