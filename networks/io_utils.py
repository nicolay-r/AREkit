import logging
import os

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.base_row import BaseRowsFormatter
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkIOUtils(object):
    """
    Provides Input/Output paths generation functions.
    """

    TERM_EMBEDDING_FILENAME = u'term_embedding'
    VOCABULARY_FILENAME = u"vocab.txt"

    @staticmethod
    def get_target_dir(experiment):
        """ Provides a main directory for input
        """
        return experiment.get_input_samples_dir()

    @staticmethod
    def get_input_opinions_filepath(experiment, data_type):
        template = NetworkIOUtils.__filename_template(data_type=data_type, experiment=experiment)
        return BaseRowsFormatter.get_filepath_static(out_dir=NetworkIOUtils.get_target_dir(experiment),
                                                     template=template,
                                                     prefix=BaseOpinionsFormatter.formatter_type_log_name())

    @staticmethod
    def get_input_sample_filepath(experiment, data_type):
        template = NetworkIOUtils.__filename_template(data_type=data_type, experiment=experiment)
        return BaseRowsFormatter.get_filepath_static(out_dir=NetworkIOUtils.get_target_dir(experiment),
                                                     template=template,
                                                     prefix=BaseSampleFormatter.formatter_type_log_name())

    @staticmethod
    def get_vocab_filepath(experiment):
        return os.path.join(NetworkIOUtils.get_target_dir(experiment),
                            NetworkIOUtils.VOCABULARY_FILENAME + u'.npz')

    @staticmethod
    def get_embedding_filepath(experiment):
        return os.path.join(NetworkIOUtils.get_target_dir(experiment),
                            NetworkIOUtils.TERM_EMBEDDING_FILENAME + u'.npz')

    @staticmethod
    def check_files_existance(data_type, experiment):
        assert(isinstance(data_type, DataType))

        filepaths = [
            NetworkIOUtils.get_input_sample_filepath(experiment=experiment, data_type=data_type),
            NetworkIOUtils.get_input_opinions_filepath(experiment=experiment, data_type=data_type),
            NetworkIOUtils.get_vocab_filepath(experiment),
            NetworkIOUtils.get_embedding_filepath(experiment)
        ]

        result = True
        for filepath in filepaths:
            existed = os.path.exists(filepath)
            logger.info("Check existance [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
            if not existed:
                result = False

        return result

    @staticmethod
    def __filename_template(data_type, experiment):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{cv_index}".format(
            data_type=data_type.name.lower(),
            cv_index=experiment.DataIO.CVFoldingAlgorithm.IterationIndex)
