from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.engine.utils import mark_dir_for_serialization
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.neutral.run import perform_neutral_annotation
from arekit.common.news.parsed.collection import ParsedNewsCollection
from arekit.contrib.bert.samplers.factory import create_bert_sample_formatter


class BertExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment,
                 skip_if_folder_exists,
                 sample_formatter_type,
                 entity_formatter,
                 write_sample_header=True):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))
        assert(isinstance(write_sample_header, bool))
        super(BertExperimentInputSerializer, self).__init__(experiment)

        self.__skip_if_folder_exists = skip_if_folder_exists
        self.__write_sample_header = write_sample_header
        self.__entity_formatter = entity_formatter
        self.__sample_formatter_type = sample_formatter_type

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        # Create samples formatter.
        sample_formatter = create_bert_sample_formatter(data_type=data_type,
                                                        formatter_type=self.__sample_formatter_type,
                                                        label_scaler=self._experiment.DataIO.LabelsScaler,
                                                        entity_formatter=self.__entity_formatter)

        # Load parsed news collections in memory.
        parsed_news_it = self._experiment.DocumentOperations.iter_parsed_news(
            self._experiment.DocumentOperations.iter_news_indices(data_type))
        parsed_news_collection = ParsedNewsCollection(parsed_news_it=parsed_news_it, notify=True)

        # Compose text opinion helper.
        # Taken from Neural networks formatter.
        text_opinion_helper = TextOpinionHelper(lambda news_id: parsed_news_collection.get_by_news_id(news_id))

        # Perform data serialization to *.tsv format.
        BaseInputEncoder.to_tsv(
            sample_filepath=self._experiment.ExperimentIO.get_input_sample_filepath(data_type=data_type),
            opinion_filepath=self._experiment.ExperimentIO.get_input_opinions_filepath(data_type=data_type),
            opinion_formatter=BaseOpinionsFormatter(data_type),
            opinion_provider=OpinionProvider.from_experiment(
                doc_ops=self._experiment.DocumentOperations,
                opin_ops=self._experiment.OpinionOperations,
                data_type=data_type,
                iter_news_ids=parsed_news_collection.iter_news_ids(),
                terms_per_context=self._experiment.DataIO.TermsPerContext,
                text_opinion_helper=text_opinion_helper),
            sample_formatter=sample_formatter,
            write_sample_header=self.__write_sample_header)

    # endregion

    # region protected methods

    def _handle_iteration(self, it_index):
        """ Performing data serialization for a particular iteration
        """
        for data_type in self._experiment.DocumentOperations.DataFolding.iter_supported_data_types():
            self.__handle_iteration(data_type)

    def _before_running(self):
        # Mark the directory as selected for serialization process.
        mark_dir_for_serialization(target_dir=self._experiment.ExperimentIO.get_target_dir(),
                                   logger=self._logger,
                                   skip_if_folder_exists=self.__skip_if_folder_exists)

        # Perform neutral annotation.
        perform_neutral_annotation(neutral_annotator=self._experiment.DataIO.NeutralAnnotator,
                                   opin_ops=self._experiment.OpinionOperations,
                                   doc_ops=self._experiment.DocumentOperations,
                                   logger=self._logger)

    # endregion