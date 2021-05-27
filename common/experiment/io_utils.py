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

    def get_experiment_sources_dir(self):
        """ Provides directory for samples.
        """
        raise NotImplementedError()

    def get_target_dir(self):
        """ Represents an experiment dir of specific label scale format,
            defined by labels scaler.
        """
        return join_dir_with_subfolder_name(subfolder_name=self.__get_experiment_folder_name(),
                                            dir=self.get_experiment_sources_dir())

    def get_experiment_folder_name(self):
        return self.__get_experiment_folder_name()

    # region protected methods

    def __get_experiment_folder_name(self):
        return u"{name}_{scale}l".format(name=self._experiment.Name,
                                         scale=str(self._experiment.DataIO.LabelsCount))

    def _experiment_iter_index(self):
        return self._experiment.DocumentOperations.DataFolding.IterationIndex

    def _filename_template(self, data_type):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                                  iter_index=self._experiment_iter_index())

    @staticmethod
    def _get_filepath(out_dir, template, prefix):
        assert(isinstance(template, unicode))
        assert(isinstance(prefix, unicode))
        return join(out_dir, BaseIOUtils.__generate_tsv_archive_filename(template=template, prefix=prefix))

    def _get_annotator_name(self):
        """ We use custom implementation as it allows to
            be independent of NeutralAnnotator instance.
        """
        return u"annot_{labels_count}l".format(labels_count=self._experiment.DataIO.LabelsCount)

    # endregion

    # region public methods

    def get_input_opinions_filepath(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.get_target_dir(),
                                  template=template,
                                  prefix=BaseOpinionsFormatter.formatter_type_log_name())

    def get_input_sample_filepath(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.get_target_dir(),
                                  template=template,
                                  prefix=BaseSampleFormatter.formatter_type_log_name())

    def create_annotated_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        annot_dir = self.__get_annotator_dir()

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        # TODO. This should not depends on the neut.
        # TODO. This should not depends on the neut.
        # TODO. This should not depends on the neut.
        filename = u"art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                           d_type=data_type.name)

        return join(annot_dir, filename)

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    # endregion

    # region private methods

    @staticmethod
    def __generate_tsv_archive_filename(template, prefix):
        return u"{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)

    def __get_annotator_dir(self):
        return join_dir_with_subfolder_name(dir=self.get_target_dir(),
                                            subfolder_name=self._get_annotator_name())

    # endregion
