from arekit.common.experiment.data_type import DataType
from arekit.common.utils import join_dir_with_subfolder_name


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def __init__(self, experiment):
        self._experiment = experiment
        self.__opinion_collection_provider = self._create_opinion_collection_provider()
        self.__opinion_collection_writer = self._create_opinion_collection_writer()

    # region abstract methods

    def get_experiment_sources_dir(self):
        """ Provides directory for samples.
        """
        raise NotImplementedError()

    def create_samples_view(self, data_type):
        raise NotImplementedError()

    def create_opinions_view(self, data_type):
        raise NotImplementedError()

    def create_samples_writer(self):
        raise NotImplementedError()

    def create_opinions_writer(self):
        raise NotImplementedError()

    def create_samples_writer_target(self, data_type):
        raise NotImplementedError()

    def create_opinions_writer_target(self, data_type):
        raise NotImplementedError()

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        raise NotImplementedError()

    def _create_annotated_collection_target(self, doc_id, data_type, check_existance):
        raise NotImplementedError()

    def _create_opinion_collection_provider(self):
        raise NotImplementedError()

    def _create_opinion_collection_writer(self):
        raise NotImplementedError()

    # endregion

    # region private methods

    def __get_experiment_folder_name(self):
        return "{name}_{scale}l".format(name=self._experiment.Name,
                                        scale=str(self._experiment.DataIO.LabelsCount))

    # endregion

    # region public methods

    def balance_samples(self, data_type, balance):
        return balance and data_type == DataType.Train

    def get_target_dir(self):
        """ Represents an experiment dir of specific label scale format,
            defined by labels scaler.
        """
        return join_dir_with_subfolder_name(subfolder_name=self.__get_experiment_folder_name(),
                                            dir=self.get_experiment_sources_dir())

    def get_experiment_folder_name(self):
        return self.__get_experiment_folder_name()

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self._create_annotated_collection_target(
            doc_id=doc_id,
            data_type=data_type,
            check_existance=check_existance)

    def write_opinion_collection(self, collection, labels_formatter, target):
        assert(target is not None)

        self.__opinion_collection_writer.serialize(
            target=target,
            collection=collection,
            labels_formatter=labels_formatter)

    def read_opinion_collection(self, target, labels_formatter, create_collection_func,
                                error_on_non_supported=False):

        # Check existance of the target.
        if target is None:
            return None

        opinions = self.__opinion_collection_provider.iter_opinions(
            source=target,
            labels_formatter=labels_formatter,
            error_on_non_supported=error_on_non_supported)

        return create_collection_func(opinions)

    # endregion
