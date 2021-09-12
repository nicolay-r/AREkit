from arekit.common.utils import join_dir_with_subfolder_name


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def __init__(self, experiment, opinion_collection_provider):
        self._experiment = experiment
        self.__opinion_collection_provider = opinion_collection_provider

    @property
    def OpinionCollectionProvider(self):
        return self.__opinion_collection_provider

    def get_experiment_sources_dir(self):
        """ Provides directory for samples.
        """
        raise NotImplementedError()

    def create_samples_reader(self, data_type):
        raise NotImplementedError()

    def create_opinions_reader(self, data_type):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def _create_annotated_collection_target(self, doc_id, data_type, check_existance):
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

    # endregion

    # region public methods

    def serialize_opinion_collection(self, collection, doc_id, data_type, labels_formatter):
        target = self._create_annotated_collection_target(
            doc_id=doc_id,
            data_type=data_type,
            check_existance=False)

        self.__opinion_collection_provider.serialize(
            target=target,
            collection=collection,
            labels_formatter=labels_formatter)

    def deserialize_opinion_collection(self, doc_id, data_type, labels_formatter, create_collection_func):
        assert(callable(create_collection_func))

        target = self._create_annotated_collection_target(
            doc_id=doc_id,
            data_type=data_type,
            check_existance=True)

        # Check existance of the target.
        if target is None:
            return None

        opinions = self.__opinion_collection_provider.iter_opinions(
            source=target,
            labels_formatter=labels_formatter,
            error_on_non_supported=False)

        return create_collection_func(opinions)

    # endregion

