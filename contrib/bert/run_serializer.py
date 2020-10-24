from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import CVBasedExperimentEngine
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.neutral.run import perform_neutral_annotation
from arekit.common.experiment.serialization_utils import mark_dir_for_serialization
from arekit.contrib.bert.core.input.io_utils import BertIOUtils
from arekit.contrib.bert.factory import create_bert_sample_formatter


class BertExperimentInputSerializer(CVBasedExperimentEngine):

    def __init__(self, experiment,
                 skip_if_folder_exists,
                 sample_formatter_type,
                 entity_formatter,
                 label_scaler,
                 write_sample_header=True,
                 io_utils=BertIOUtils):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))
        # TODO. Move to data_io (after data_io refactoring)
        assert(isinstance(write_sample_header, bool))
        super(BertExperimentInputSerializer, self).__init__(experiment)

        self.__skip_if_folder_exists = skip_if_folder_exists
        self.__io_utils = io_utils
        # TODO. Move to data_io (after data_io refactoring)
        self.__write_sample_header = write_sample_header
        self.__entity_formatter = entity_formatter
        self.__sample_formatter_type = sample_formatter_type
        self.__label_scaler = label_scaler

    # region private methods

    def __handle_cv_index(self, data_type):
        assert(isinstance(data_type, DataType))

        # Create samples formatter.
        sample_formatter = create_bert_sample_formatter(data_type=data_type,
                                                        formatter_type=self.__sample_formatter_type,
                                                        label_scaler=self.__label_scaler,
                                                        entity_formatter=self.__entity_formatter)

        # Load parsed news collections in memory.
        # Taken from Neural networks formatter.
        parsed_news_collection = self._experiment.create_parsed_collection(data_type)

        # Compose text opinion helper.
        # Taken from Neural networks formatter.
        text_opinion_helper = TextOpinionHelper(lambda news_id: parsed_news_collection.get_by_news_id(news_id))

        # Perform data serialization to *.tsv format.
        BaseInputEncoder.to_tsv(
            sample_filepath=BertIOUtils.get_input_sample_filepath(experiment=self._experiment, data_type=data_type),
            opinion_filepath=BertIOUtils.get_input_opinions_filepath(experiment=self._experiment, data_type=data_type),
            opinion_formatter=BaseOpinionsFormatter(data_type),
            opinion_provider=OpinionProvider.from_experiment(
                experiment=self._experiment,
                data_type=data_type,
                iter_news_ids=parsed_news_collection.iter_news_ids(),
                terms_per_context=self._experiment.DataIO.TermsPerContext,
                text_opinion_helper=text_opinion_helper),
            sample_formatter=sample_formatter,
            write_sample_header=self.__write_sample_header)

    # endregion

    # region protected methods

    def _handle_cv_index(self, cv_index):
        """ Performing data serialization for a particular cv_index
        cv_index: int
        """
        for data_type in self._experiment.DocumentOperations.iter_supported_data_types():
            self.__handle_cv_index(data_type)

    def _before_running(self):
        # Mark the directory as selected for serialization process.
        mark_dir_for_serialization(experiment=self._experiment,
                                   logger=self._logger,
                                   io_utils=self.__io_utils,
                                   skip_if_folder_exists=self.__skip_if_folder_exists)

        # Perform neutral annotation.
        perform_neutral_annotation(experiment=self._experiment,
                                   logger=self._logger)

    # endregion