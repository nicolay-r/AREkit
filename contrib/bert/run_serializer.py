from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.neutral.run import perform_neutral_annotation
from arekit.contrib.bert.samplers.factory import create_bert_sample_formatter


class BertExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment,
                 skip_if_folder_exists,
                 sample_formatter_type,
                 entity_formatter,
                 balance_train_samples,
                 write_sample_header=True):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))
        assert(isinstance(write_sample_header, bool))
        super(BertExperimentInputSerializer, self).__init__(experiment)

        self.__skip_if_folder_exists = skip_if_folder_exists
        self.__write_sample_header = write_sample_header
        self.__entity_formatter = entity_formatter
        self.__sample_formatter_type = sample_formatter_type
        self.__balance_train_samples = balance_train_samples

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        # Create samples formatter.
        sample_formatter = create_bert_sample_formatter(
            data_type=data_type,
            formatter_type=self.__sample_formatter_type,
            label_scaler=self._experiment.DataIO.LabelsScaler,
            entity_formatter=self.__entity_formatter,
            balance=self.__balance_train_samples,
            entity_to_group_func=self._experiment.entity_to_group)

        # Perform data serialization to *.tsv format.
        BaseInputEncoder.to_tsv(
            sample_filepath=self._experiment.ExperimentIO.get_input_sample_filepath(data_type=data_type),
            opinion_filepath=self._experiment.ExperimentIO.get_input_opinions_filepath(data_type=data_type),
            opinion_formatter=BaseOpinionsFormatter(data_type),
            opinion_provider=OpinionProvider.from_experiment(
                doc_ops=self._experiment.DocumentOperations,
                opin_ops=self._experiment.OpinionOperations,
                data_type=data_type,
                parsed_news_it_func=lambda: self.__iter_parsed_news(
                    doc_ops=self._experiment.DocumentOperations,
                    data_type=data_type),
                terms_per_context=self._experiment.DataIO.TermsPerContext),
            sample_formatter=sample_formatter,
            write_sample_header=self.__write_sample_header)

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
        perform_neutral_annotation(neutral_annotator=self._experiment.DataIO.NeutralAnnotator,
                                   opin_ops=self._experiment.OpinionOperations,
                                   doc_ops=self._experiment.DocumentOperations,
                                   logger=self._logger)

    # endregion