from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.experiment.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.experiment.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.experiment.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.samplers.factory import create_bert_sample_provider


class BertExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment,
                 labels_formatter,
                 skip_if_folder_exists,
                 sample_formatter_type,
                 entity_formatter,
                 balance_train_samples):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        super(BertExperimentInputSerializer, self).__init__(experiment)

        self.__skip_if_folder_exists = skip_if_folder_exists
        self.__entity_formatter = entity_formatter
        self.__sample_formatter_type = sample_formatter_type
        self.__balance_train_samples = balance_train_samples
        self.__labels_formatter = labels_formatter

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))
        opinions_target = self._experiment.ExperimentIO.create_opinions_writer_target(data_type)
        samples_target = self._experiment.ExperimentIO.create_samples_writer_target(data_type)
        opinions_storage = self._experiment.ExperimentIO.create_opinions_writer(data_type)
        samples_storage = self._experiment.ExperimentIO.create_samples_writer(
            data_type=data_type,
            balance=self.__balance_train_samples)

        # Create samples formatter.
        sample_rows_provider = create_bert_sample_provider(
            labels_formatter=self.__labels_formatter,
            provider_type=self.__sample_formatter_type,
            label_scaler=self._experiment.DataIO.LabelsScaler,
            entity_formatter=self.__entity_formatter,
            entity_to_group_func=self._experiment.entity_to_group)

        # Create repositories
        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=opinions_storage)
        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=sample_rows_provider,
            storage=samples_storage)

        # Create opinion provider
        opinion_provider = OpinionProvider.create(
            read_news_func=lambda news_id: self._experiment.DocumentOperations.read_news(news_id),
            iter_news_opins_for_extraction=lambda news_id:
                self._experiment.OpinionOperations.iter_opinions_for_extraction(
                    doc_id=news_id,
                    data_type=data_type),
            parsed_news_it_func=lambda: self.__iter_parsed_news(
                doc_ops=self._experiment.DocumentOperations,
                data_type=data_type),
            terms_per_context=self._experiment.DataIO.TermsPerContext)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               target=opinions_target,
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              target=samples_target,
                              desc="sample")

    @staticmethod
    def __iter_parsed_news(doc_ops, data_type):
        return doc_ops.iter_parsed_news(doc_ops.iter_doc_ids(data_type))

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