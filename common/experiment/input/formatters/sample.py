from collections import OrderedDict

from arekit.bert.input.providers.row_ids.binary import BinaryIDProvider
from arekit.bert.input.providers.label.binary import BinaryLabelProvider
from arekit.common.experiment.input.formatters.base_row import BaseRowsFormatter
from arekit.common.experiment.input.providers.label.base import LabelProvider
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import Label
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.experiment.formats.base import BaseExperiment


class BaseSampleFormatter(BaseRowsFormatter):
    """
    Custom Processor with the following fields

    [id, label, text_a] -- for train
    [id, text_a] -- for test
    """

    """
    Fields
    """
    ID = 'id'
    LABEL = 'label'
    S_IND = 's_ind'
    T_IND = 't_ind'

    def __init__(self, data_type, label_provider, text_provider):
        assert(isinstance(label_provider, LabelProvider))
        assert(isinstance(text_provider, BaseSingleTextProvider))

        self.__label_provider = label_provider
        self.__text_provider = text_provider
        self.__row_ids_provider = self.__create_row_ids_provider(label_provider)

        super(BaseSampleFormatter, self).__init__(data_type=data_type)

    @staticmethod
    def formatter_type_log_name():
        return u"sample"

    # region Private methods

    @staticmethod
    def __create_row_ids_provider(label_provider):
        if isinstance(label_provider, BinaryLabelProvider):
            return BinaryIDProvider()
        if isinstance(label_provider, MultipleLabelProvider):
            return MultipleIDProvider()

    def __is_train(self):
        return self._data_type == DataType.Train

    def _get_columns_list_with_types(self):
        """
        Composing df with the following columns:
            [id, label, type, text_a]
        """
        dtypes_list = super(BaseSampleFormatter, self)._get_columns_list_with_types()

        dtypes_list.append((self.ID, unicode))

        # insert labels
        if self.__is_train():
            dtypes_list.append((self.LABEL, 'int32'))

        # insert text columns
        for col_name in self.__text_provider.iter_columns():
            dtypes_list.append((col_name, unicode))

        # insert indices
        dtypes_list.append((self.S_IND, 'int32'))
        dtypes_list.append((self.T_IND, 'int32'))

        return dtypes_list

    @staticmethod
    def __get_opinion_end_indices(parsed_news, text_opinion):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind = parsed_news.get_entity_position(text_opinion.SourceId).get_index(
            position_type=TermPositionTypes.IndexInSentence)

        t_ind = parsed_news.get_entity_position(text_opinion.TargetId).get_index(
            position_type=TermPositionTypes.IndexInSentence)

        return (s_ind, t_ind)

    def _fill_row_core(self, row, opinion_provider, linked_wrap, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):

        row[self.ID] = self.__row_ids_provider.create_sample_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_wrap,
            index_in_linked=index_in_linked,
            label_scaler=self.__label_provider.LabelScaler)

        expected_label = linked_wrap.get_linked_label()

        if self.__is_train():
            row[self.LABEL] = self.__label_provider.calculate_output_label(
                expected_label=expected_label,
                etalon_label=etalon_label)

        terms = list(parsed_news.iter_sentence_terms(sentence_index=sentence_ind,
                                                     return_id=False))
        self.__text_provider.add_text_in_row(row=row,
                                             sentence_terms=terms,
                                             s_ind=s_ind,
                                             t_ind=t_ind,
                                             expected_label=expected_label)

        row[self.S_IND] = s_ind
        row[self.T_IND] = t_ind

    def __create_row(self, opinion_provider, linked_wrap, index_in_linked, etalon_label):
        """
        Composing row in following format:
            [id, label, type, text_a]

        returns: OrderedDict
            row with key values
        """
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(etalon_label, Label))

        text_opinion = linked_wrap[index_in_linked]

        parsed_news, sentence_ind = opinion_provider.get_opinion_location(text_opinion)
        s_ind, t_ind = self.__get_opinion_end_indices(parsed_news, text_opinion)

        row = OrderedDict()

        self._fill_row_core(row=row,
                            parsed_news=parsed_news,
                            sentence_ind=sentence_ind,
                            opinion_provider=opinion_provider,
                            linked_wrap=linked_wrap,
                            index_in_linked=index_in_linked,
                            etalon_label=etalon_label,
                            s_ind=s_ind,
                            t_ind=t_ind)
        return row

    def __provide_rows(self, opinion_provider, linked_wrap, index_in_linked):
        """
        Providing Rows depending on row_id_formatter type
        """
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))

        origin = linked_wrap.First
        if isinstance(self.__row_ids_provider, BinaryIDProvider):
            """
            Enumerate all opinions as if it would be with the different label types.
            """
            for label in self.__label_provider.SupportedLabels:
                yield self.__create_row(opinion_provider=opinion_provider,
                                        linked_wrap=self.__copy_modified_linked_wrap(linked_wrap, label),
                                        index_in_linked=index_in_linked,
                                        etalon_label=origin.Sentiment)

        if isinstance(self.__row_ids_provider, MultipleIDProvider):
            yield self.__create_row(opinion_provider=opinion_provider,
                                    linked_wrap=linked_wrap,
                                    index_in_linked=index_in_linked,
                                    etalon_label=origin.Sentiment)

    @staticmethod
    def __copy_modified_linked_wrap(linked_wrap, label):
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))
        linked_opinions = [o for o in linked_wrap]

        copy = TextOpinion.create_copy(other=linked_opinions[0])
        copy.set_label(label=label)

        linked_opinions[0] = copy

        return LinkedTextOpinionsWrapper(linked_text_opinions=linked_opinions)

    def _iter_by_rows(self, opinion_provider):
        """
        Iterate by rows that is assumes to be added as samples, using opinion_provider information
        """
        assert(isinstance(opinion_provider, OpinionProvider))

        linked_iter = opinion_provider.iter_linked_opinion_wrappers(
            balance=self.__is_train(),
            supported_labels=self.__label_provider.SupportedLabels)

        for linked_wrap in linked_iter:

            for i in range(len(linked_wrap)):

                rows_it = self.__provide_rows(
                    opinion_provider=opinion_provider,
                    linked_wrap=linked_wrap,
                    index_in_linked=i)

                for row in rows_it:
                    yield row

    # endregion

    def to_tsv_by_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))

        filepath = self.get_filepath(data_type=self._data_type,
                                     experiment=experiment)

        # TODO. This should be in different function.
        self._df.to_csv(filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        float_format="%.0f",
                        compression='gzip',
                        header=not self.__is_train())

    def __len__(self):
        return len(self._df.index)
