import logging
from os.path import join

from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.input.writers.factory import create_writer_extension
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import LinkedSamplesStorageView
from arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arekit.contrib.utils.io_utils.utils import filename_template, check_targets_existence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SamplesIO(BaseSamplesIO):
    """ Samples default IO utils for samples.
            Sample is a text part which include pair of attitude participants.
            This class allows to provide saver and loader for such entries, bubbed as samples.
            Samples required for machine learning training/inferring.
    """

    def __init__(self, target_dir, writer, prefix="sample", target_extension=None):
        assert(isinstance(target_dir, str))
        assert(isinstance(writer, BaseWriter))
        assert(isinstance(prefix, str))
        assert(isinstance(target_extension, str) or target_extension is None)
        self.__target_dir = target_dir
        self.__writer = writer
        self.__prefix = prefix
        self.__target_extension = create_writer_extension(writer) if target_extension is None else target_extension

    # region public methods

    def create_view(self, target):
        return LinkedSamplesStorageView(storage=BaseRowsStorage.from_tsv(filepath=target),
                                        row_ids_provider=MultipleIDProvider())

    def create_writer(self):
        return self.__writer

    def create_target(self, data_type, data_folding):
        return self.__get_input_sample_target(data_type, data_folding=data_folding)

    def check_targets_existed(self, data_types_iter, data_folding):
        for data_type in data_types_iter:

            targets = [
                self.__get_input_sample_target(data_type=data_type, data_folding=data_folding),
            ]

            if not check_targets_existence(targets=targets):
                return False
        return True

    # endregion

    def __get_input_sample_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__target_dir,
                                   template=template,
                                   prefix=self.__prefix,
                                   extension=self.__target_extension)

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
