import logging
import os
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.io_utils import BaseIOUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkIOUtils(BaseIOUtils):
    """ Provides additional Input/Output paths generation functions for:
        - embedding matrix;
        - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = u'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = u"vocab-{cv_index}.txt"

    @classmethod
    def get_vocab_filepath(cls, experiment):
        return os.path.join(cls.get_target_dir(experiment),
                            cls.VOCABULARY_FILENAME_TEMPLATE.format(cv_index=NetworkIOUtils._get_cv_index(experiment)) + u'.npz')

    @classmethod
    def get_embedding_filepath(cls, experiment):
        return os.path.join(cls.get_target_dir(experiment),
                            cls.TERM_EMBEDDING_FILENAME_TEMPLATE.format(cv_index=NetworkIOUtils._get_cv_index(experiment)) + u'.npz')

    @classmethod
    def check_files_existance(cls, data_type, experiment):
        assert(isinstance(data_type, DataType))

        filepaths = [
            cls.get_input_sample_filepath(experiment=experiment, data_type=data_type),
            cls.get_input_opinions_filepath(experiment=experiment, data_type=data_type),
            cls.get_vocab_filepath(experiment),
            cls.get_embedding_filepath(experiment)
        ]

        result = True
        for filepath in filepaths:
            existed = os.path.exists(filepath)
            logger.info("Check existance [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
            if not existed:
                result = False

        return result