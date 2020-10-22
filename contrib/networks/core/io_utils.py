import logging
import os

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkIOUtils(object):
    """
    Provides Input/Output paths generation functions.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = u'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = u"vocab-{cv_index}.txt"

    @staticmethod
    def get_target_dir(experiment):
        """ Provides a main directory for input
        """
        return experiment.DataIO.get_input_samples_dir(experiment.Name)

    @classmethod
    def get_input_opinions_filepath(cls, experiment, data_type):
        template = NetworkIOUtils.__filename_template(data_type=data_type,
                                                      experiment=experiment)
        return cls.__get_filepath(out_dir=cls.get_target_dir(experiment),
                                  template=template,
                                  prefix=BaseOpinionsFormatter.formatter_type_log_name())

    @classmethod
    def get_input_sample_filepath(cls, experiment, data_type):
        template = cls.__filename_template(data_type=data_type, experiment=experiment)
        return cls.__get_filepath(out_dir=cls.get_target_dir(experiment),
                                  template=template,
                                  prefix=BaseSampleFormatter.formatter_type_log_name())

    @classmethod
    def get_output_results_filepath(cls, experiment, data_type, epoch_index):
        f_name_template = cls.__filename_template(data_type=data_type,
                                                  experiment=experiment)

        result_template = u"".join([f_name_template, u'-e{e_index}'.format(e_index=epoch_index)])

        return cls.__get_filepath(out_dir=experiment.DataIO.ModelIO.ModelRoot,
                                  template=result_template,
                                  prefix=u"result")

    @classmethod
    def get_vocab_filepath(cls, experiment):
        return os.path.join(cls.get_target_dir(experiment),
                            cls.VOCABULARY_FILENAME_TEMPLATE.format(cv_index=NetworkIOUtils.__get_cv_index(experiment)) + u'.npz')

    @classmethod
    def get_embedding_filepath(cls, experiment):
        return os.path.join(cls.get_target_dir(experiment),
                            cls.TERM_EMBEDDING_FILENAME_TEMPLATE.format(cv_index=NetworkIOUtils.__get_cv_index(experiment)) + u'.npz')

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

    # region private methods

    @staticmethod
    def __filename_template(data_type, experiment):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{cv_index}".format(data_type=data_type.name.lower(),
                                                cv_index=NetworkIOUtils.__get_cv_index(experiment))

    @staticmethod
    def __get_cv_index(experiment):
        return experiment.DataIO.CVFoldingAlgorithm.IterationIndex

    @staticmethod
    def __get_filepath(out_dir, template, prefix):
        assert(isinstance(template, unicode))
        assert(isinstance(prefix, unicode))
        filepath = os.path.join(out_dir, NetworkIOUtils.__generate_filename(template=template, prefix=prefix))
        return filepath

    @staticmethod
    def __generate_filename(template, prefix):
        return u"{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

    # endregion