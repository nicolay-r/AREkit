import logging
import os
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.model.model_io import BaseModelIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkIOUtils(BaseIOUtils):
    """ Provides additional Input/Output paths generation functions for:
        - model directory;
        - embedding matrix;
        - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = u'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = u"vocab-{cv_index}.txt"

    @classmethod
    def get_target_dir(cls, experiment):
        """ Represents an experiment dir of specific label scale format,
            defined by labels scaler.
        """
        e_name = u"{name}_{scale}l".format(name=experiment.Name,
                                           scale=experiment.LabelsScaler.LabelsCount)

        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=e_name,
            experiments_dir=experiment.DataIO.get_experiment_sources_dir())

    @classmethod
    def get_vocab_filepath(cls, experiment):
        return os.path.join(cls.get_target_dir(experiment),
                            cls.VOCABULARY_FILENAME_TEMPLATE.format(
                                cv_index=NetworkIOUtils._get_cv_index(experiment)) + u'.npz')

    @classmethod
    def get_embedding_filepath(cls, experiment):
        return os.path.join(cls.get_target_dir(experiment),
                            cls.TERM_EMBEDDING_FILENAME_TEMPLATE.format(
                                cv_index=NetworkIOUtils._get_cv_index(experiment)) + u'.npz')

    @classmethod
    def get_output_model_results_filepath(cls, experiment, data_type, epoch_index):

        f_name_template = cls.__filename_template(data_type=data_type,
                                                  experiment=experiment)

        result_template = u"".join([f_name_template, u'-e{e_index}'.format(e_index=epoch_index)])

        # Perform access to the model, since all the IO information
        # that is related to the model, assumes to be stored in ModelIO.
        model_io = experiment.DataIO.ModelIO
        assert(isinstance(model_io, BaseModelIO))

        return cls.__get_filepath(out_dir=model_io.get_model_dir(),
                                  template=result_template,
                                  prefix=u"result")

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
