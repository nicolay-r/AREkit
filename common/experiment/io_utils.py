from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    @classmethod
    def get_target_dir(cls, experiment):
        """ Provides a main directory for input
        """
        return experiment.DataIO.get_input_samples_dir(experiment.Name)

    @classmethod
    def get_input_opinions_filepath(cls, experiment, data_type):
        template = cls.__filename_template(data_type=data_type,
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

    @staticmethod
    def _get_cv_index(experiment):
        return experiment.DataIO.CVFoldingAlgorithm.IterationIndex

    # region private methods

    @staticmethod
    def __filename_template(data_type, experiment):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{cv_index}".format(data_type=data_type.name.lower(),
                                                cv_index=BaseIOUtils._get_cv_index(experiment))

    @staticmethod
    def __get_filepath(out_dir, template, prefix):
        assert(isinstance(template, unicode))
        assert(isinstance(prefix, unicode))
        filepath = join(out_dir, BaseIOUtils.__generate_filename(template=template, prefix=prefix))
        return filepath

    @staticmethod
    def __generate_filename(template, prefix):
        return u"{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

    # endregion
