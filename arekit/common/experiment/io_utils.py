from os.path import join

from arekit.common.experiment.data_type import DataType
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

    def create_samples_reader(self, data_type):
        raise NotImplementedError()

    def create_opinions_reader(self, data_type):
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
        return "{name}_{scale}l".format(name=self._experiment.Name,
                                        scale=str(self._experiment.DataIO.LabelsCount))

    def __get_annotator_dir(self):
        return join_dir_with_subfolder_name(dir=self.get_target_dir(),
                                            subfolder_name=self._get_annotator_name())

    def _get_annotator_name(self):
        """ We use custom implementation as it allows to
            be independent of NeutralAnnotator instance.
        """
        return "annot_{labels_count}l".format(labels_count=self._experiment.DataIO.LabelsCount)

    # endregion

    # region public methods

    def create_annotated_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        annot_dir = self.__get_annotator_dir()

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        # TODO. This should not depends on the neut.
        # TODO. This should not depends on the neut.
        # TODO. This should not depends on the neut.
        filename = "art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                          d_type=data_type.name)

        return join(annot_dir, filename)

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    # endregion

