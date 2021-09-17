from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.provider import BaseInputProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.experiment.input.storages.tsv_opinion import TsvOpinionsStorage
from arekit.common.experiment.input.storages.tsv_sample import TsvSampleStorage
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.samplers.factory import create_bert_sample_provider


class BertExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment,
                 labels_formatter,
                 skip_if_folder_exists,
                 sample_formatter_type,
                 entity_formatter,
                 balance_train_samples,
                 write_sample_header=True):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))
        assert(isinstance(write_sample_header, bool))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        super(BertExperimentInputSerializer, self).__init__(experiment)

        self.__skip_if_folder_exists = skip_if_folder_exists
        self.__write_sample_header = write_sample_header
        self.__entity_formatter = entity_formatter
        self.__sample_formatter_type = sample_formatter_type
        self.__balance_train_samples = balance_train_samples
        self.__labels_formatter = labels_formatter

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        sample_storage = TsvSampleStorage(
            store_labels=data_type == DataType.Train,
            balance=self.__balance_train_samples,
            write_header=self.__write_sample_header,
            filepath=self._experiment.ExperimentIO.get_input_sample_filepath(
                data_type=data_type))

        opinions_storage = TsvOpinionsStorage(
            filepath=self._experiment.ExperimentIO.get_input_opinions_filepath(data_type=data_type))

        # Create samples formatter.
        sample_rows_provider = create_bert_sample_provider(
            storage=sample_storage,
            labels_formatter=self.__labels_formatter,
            provider_type=self.__sample_formatter_type,
            label_scaler=self._experiment.DataIO.LabelsScaler,
            entity_formatter=self.__entity_formatter,
            entity_to_group_func=self._experiment.entity_to_group)

        opinions_row_provider = BaseOpinionsRowProvider(storage=opinions_storage)

        # Perform data serialization to *.tsv format.
        BaseInputProvider.save(
            sample_row_provider=sample_rows_provider,
            opinion_row_provider=opinions_row_provider,
            opinion_provider=OpinionProvider.from_experiment(
                doc_ops=self._experiment.DocumentOperations,
                opin_ops=self._experiment.OpinionOperations,
                data_type=data_type,
                parsed_news_it_func=lambda: self.__iter_parsed_news(
                    doc_ops=self._experiment.DocumentOperations,
                    data_type=data_type),
                terms_per_context=self._experiment.DataIO.TermsPerContext))

    @staticmethod
    def __iter_parsed_news(doc_ops, data_type):
        return doc_ops.iter_parsed_news(doc_ops.iter_news_indices(data_type))

    # endregion

    # region protected methods

    def _handle_iteration(self, it_index):
        """ Performing data serialization for a particular iteration
        """
        for data_type in self._experiment.DocumentOperations.DataFolding.iter_supported_data_types():
            self.__handle_iteration(data_type)

    def _before_running(self):
        self._logger.info("Perform annotation ...")
        for data_type in self._experiment.DocumentOperations.DataFolding.iter_supported_data_types():
            self._experiment.DataIO.Annotator.serialize_missed_collections(
                data_type=data_type,
                opin_ops=self._experiment.OpinionOperations,
                doc_ops=self._experiment.DocumentOperations)

    # endregion