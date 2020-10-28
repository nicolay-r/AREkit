from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir


# TODO. Make this non static
class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    # TODO. Init as follows.
    def __init__(self, experiment):
        self.__experiment = experiment

    @classmethod
    def get_target_dir(cls, experiment):
        """ Provides a main directory for input
            Assumes to be manually implemented for every nested base_io utils.
        """
        raise NotImplementedError()

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

    @classmethod
    def create_neutral_opinion_collection_filepath(cls, experiment, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        annot_dir = cls.__get_neutral_annotation_dir(experiment)

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        filename = u"art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                           d_type=data_type.name)

        return join(annot_dir, filename)

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

    @classmethod
    def __get_neutral_annotation_dir(cls, experiment):
        return get_path_of_subfolder_in_experiments_dir(experiments_dir=cls.get_target_dir(experiment),
                                                        subfolder_name=experiment.DataIO.NeutralAnnotator.Name)

    # endregion
