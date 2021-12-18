from arekit.common.data.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.opinions import OpinionProvider
from arekit.common.data.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.data.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.samplers.factory import create_bert_sample_provider


class BertExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment,
                 labels_formatter,
                 skip_if_folder_exists,
                 sample_provider_type,
                 entity_formatter,
                 balance_train_samples):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        super(BertExperimentInputSerializer, self).__init__(experiment)

        self.__skip_if_folder_exists = skip_if_folder_exists
        self.__entity_formatter = entity_formatter
        self.__sample_provider_type = sample_provider_type
        self.__balance_train_samples = balance_train_samples
        self.__labels_formatter = labels_formatter

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        # Create samples formatter.
        sample_rows_provider = create_bert_sample_provider(
            labels_formatter=self.__labels_formatter,
            provider_type=self.__sample_provider_type,
            label_scaler=self._experiment.DataIO.LabelsScaler,
            entity_formatter=self.__entity_formatter,
            entity_to_group_func=self._experiment.entity_to_group)

        # Create repositories
        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage())
        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=sample_rows_provider,
            storage=BaseRowsStorage())

        # Create opinion provider
        opinion_provider = OpinionProvider.create(
            value_to_group_id_func=None,
            parse_news_func=lambda doc_id: self._experiment.DocumentOperations.parse_doc(doc_id),
            iter_news_opins_for_extraction=lambda doc_id:
                self._experiment.OpinionOperations.iter_opinions_for_extraction(doc_id=doc_id, data_type=data_type),
            terms_per_context=self._experiment.DataIO.TermsPerContext)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               doc_ids_iter=self._experiment.DocumentOperations.iter_doc_ids(data_type),
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              doc_ids_iter=self._experiment.DocumentOperations.iter_doc_ids(data_type),
                              desc="sample")

        if self._experiment.ExperimentIO.balance_samples(data_type=data_type, balance=self.__balance_train_samples):
            samples_repo.balance()

        # Save repositories
        samples_repo.write(
            target=self._experiment.ExperimentIO.create_samples_writer_target(data_type),
            writer=self._experiment.ExperimentIO.create_samples_writer())

        opinions_repo.write(
            target=self._experiment.ExperimentIO.create_opinions_writer_target(data_type),
            writer=self._experiment.ExperimentIO.create_opinions_writer())

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

            collections_it = self._experiment.DataIO.Annotator.iter_annotated_collections(
                data_type=data_type,
                opin_ops=self._experiment.OpinionOperations,
                doc_ops=self._experiment.DocumentOperations)

            for doc_id, collection in collections_it:

                target = self._experiment.ExperimentIO.create_opinion_collection_target(
                    doc_id=doc_id,
                    data_type=data_type)

                self._experiment.write_opinion_collection(
                    collection=collection,
                    target=target,
                    labels_formatter=self.__labels_formatter)

    # endregion