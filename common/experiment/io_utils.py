from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.utils import join_dir_with_subfolder_name


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def __init__(self, experiment):
        self._experiment = experiment

    def get_target_dir(self):
        """ Provides a main directory for input
            Assumes to be manually implemented for every nested base_io utils.
        """
        raise NotImplementedError()

    # region protected methods

    def _get_cv_index(self):
        return self._experiment.DataIO.CVFoldingAlgorithm.IterationIndex

    # endregion

    # region public methods

    def get_input_opinions_filepath(self, data_type):
        template = self.__filename_template(data_type=data_type)
        return self.__get_filepath(out_dir=self.get_target_dir(),
                                   template=template,
                                   prefix=BaseOpinionsFormatter.formatter_type_log_name())

    def get_input_sample_filepath(self, data_type):
        template = self.__filename_template(data_type=data_type)
        return self.__get_filepath(out_dir=self.get_target_dir(),
                                   template=template,
                                   prefix=BaseSampleFormatter.formatter_type_log_name())

    def create_neutral_opinion_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        annot_dir = self.__get_neutral_annotation_dir()

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        filename = u"art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                           d_type=data_type.name)

        return join(annot_dir, filename)

    # endregion

    # region private methods

    def __filename_template(self, data_type):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{cv_index}".format(data_type=data_type.name.lower(),
                                                cv_index=self._get_cv_index())

    @staticmethod
    def __get_filepath(out_dir, template, prefix):
        assert(isinstance(template, unicode))
        assert(isinstance(prefix, unicode))
        filepath = join(out_dir, BaseIOUtils.__generate_filename(template=template, prefix=prefix))
        return filepath

    @staticmethod
    def __generate_filename(template, prefix):
        return u"{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

    def __get_neutral_annotation_dir(self):
        return join_dir_with_subfolder_name(dir=self.get_target_dir(),
                                            subfolder_name=self._experiment.DataIO.NeutralAnnotator.Name)

    # endregion
