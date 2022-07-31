import logging
from os.path import join

from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView
from arekit.contrib.utils.io_utils.utils import filename_template, check_targets_existence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SamplesIOUtils(BaseIOUtils):
    """ Samples default IO utils for samples.
            Sample is a text part which include pair of attitude participants.
            This class allows to provide saver and loader for such entries, bubbed as samples.
            Samples required for machine learning training/inferring.
    """

    def __init__(self, target_dir,
                 samples_writer=TsvWriter(write_header=True),
                 target_extension=".tsv.gz"):
        assert(isinstance(samples_writer, BaseWriter))
        self.__target_dir = target_dir
        self.__samples_writer = samples_writer
        self.__target_extension = target_extension

    # region public methods

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
        return self.__samples_writer

    def create_samples_writer(self):
        return self.__samples_writer

    def create_target_extension(self):
        return self.__target_extension

    def create_opinions_writer_target(self, data_type, data_folding):
        return self.__get_input_opinions_target(data_type, data_folding=data_folding)

    def create_samples_writer_target(self, data_type, data_folding):
        return self.__get_input_sample_target(data_type, data_folding=data_folding)

    def check_targets_existed(self, data_types_iter, data_folding):
        for data_type in data_types_iter:

            targets = [
                self.__get_input_sample_target(data_type=data_type, data_folding=data_folding),
                # self.__get_input_opinions_target(data_type=data_type, data_folding=data_folding),
            ]

            if not check_targets_existence(targets=targets):
                return False
        return True

    # endregion

    def __get_input_opinions_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__target_dir,
                                   template=template,
                                   prefix="opinion",
                                   extension=self.create_target_extension())

    def __get_input_sample_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__target_dir,
                                   template=template,
                                   prefix="sample",
                                   extension=self.create_target_extension())

    # endregion

    # region protected methods

    @staticmethod
    def __get_filepath(out_dir, template, prefix, extension):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        assert(isinstance(extension, str))
        return join(out_dir, "{prefix}-{template}{extension}".format(
            prefix=prefix, template=template, extension=extension))

    # endregion
