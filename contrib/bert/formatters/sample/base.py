import logging
import random
import pandas as pd
from collections import OrderedDict

from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import Label
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.experiment.formats.base import BaseExperiment

from arekit.contrib.bert.providers.label.base import BertLabelProvider
from arekit.contrib.bert.providers.label.binary import BertBinaryLabelProvider
from arekit.contrib.bert.providers.label.multiple import BertMultipleLabelProvider
from arekit.contrib.bert.providers.row_ids.binary import BinaryIDProvider
from arekit.contrib.bert.providers.row_ids.multiple import MultipleIDProvider
from arekit.contrib.bert.providers.text.single import SingleTextProvider

from arekit.contrib.bert.formatters.base import BaseBertRowsFormatter
from arekit.contrib.bert.providers.opinions import OpinionProvider


logger = logging.getLogger(__name__)


class BaseSampleFormatter(BaseBertRowsFormatter):
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
        assert(isinstance(label_provider, BertLabelProvider))
        assert(isinstance(text_provider, SingleTextProvider))

        self.__label_provider = label_provider
        self.__text_provider = text_provider
        self.__row_ids_formatter = self.__create_row_ids_formatter(label_provider)

        super(BaseSampleFormatter, self).__init__(data_type=data_type)

    @staticmethod
    def formatter_type_log_name():
        return u"sample"

    # region Private methods

    @staticmethod
    def __create_row_ids_formatter(label_provider):
        if isinstance(label_provider, BertBinaryLabelProvider):
            return BinaryIDProvider()
        if isinstance(label_provider, BertMultipleLabelProvider):
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

        s_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.SourceId)
        t_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.TargetId)

        return (s_ind, t_ind)

    def __create_row(self, opinion_provider, linked_wrap, index_in_linked, etalon_label, idle_mode):
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
        assert(isinstance(idle_mode, bool))

        if idle_mode:
            return None

        text_opinion = linked_wrap[index_in_linked]

        parsed_news, sentence_ind = opinion_provider.get_opinion_location(text_opinion)
        s_ind, t_ind = self.__get_opinion_end_indices(parsed_news, text_opinion)

        row = OrderedDict()

        row[self.ID] = self.__row_ids_formatter.create_sample_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_wrap,
            index_in_linked=index_in_linked,
            label_scaler=self.__label_provider.LabelScaler)

        expected_label = linked_wrap.get_linked_label()

        if self.__is_train():
            row[self.LABEL] = self.__label_provider.calculate_output_label_uint(
                expected_label=expected_label,
                etalon_label=etalon_label)

        terms = list(parsed_news.iter_sentence_terms(sentence_ind))
        self.__text_provider.add_text_in_row(row=row,
                                             sentence_terms=terms,
                                             s_ind=s_ind,
                                             t_ind=t_ind,
                                             expected_label=expected_label)

        row[self.S_IND] = s_ind
        row[self.T_IND] = t_ind

        return row

    def __provide_rows(self, opinion_provider, linked_wrap, index_in_linked, idle_mode):
        """
        Providing Rows depending on row_id_formatter type
        """
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))

        origin = linked_wrap.First
        if isinstance(self.__row_ids_formatter, BinaryIDProvider):
            """
            Enumerate all opinions as if it would be with the different label types.
            """
            for label in self.__label_provider.SupportedLabels:
                yield self.__create_row(opinion_provider=opinion_provider,
                                        linked_wrap=self.__copy_modified_linked_wrap(linked_wrap, label),
                                        index_in_linked=index_in_linked,
                                        etalon_label=origin.Sentiment,
                                        idle_mode=idle_mode)

        if isinstance(self.__row_ids_formatter, MultipleIDProvider):
            yield self.__create_row(opinion_provider=opinion_provider,
                                    linked_wrap=linked_wrap,
                                    index_in_linked=index_in_linked,
                                    etalon_label=origin.Sentiment,
                                    idle_mode=idle_mode)

    @staticmethod
    def __copy_modified_linked_wrap(linked_wrap, label):
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))
        linked_opinions = [o for o in linked_wrap]

        copy = TextOpinion.create_copy(other=linked_opinions[0])
        copy.set_label(label=label)

        linked_opinions[0] = copy

        return LinkedTextOpinionsWrapper(linked_text_opinions=linked_opinions)

    def _iter_by_rows(self, opinion_provider, idle_mode):
        """
        Iterate by rows that is assumes to be added as samples, using opinion_provider information
        """
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(idle_mode, bool))

        linked_iter = opinion_provider.iter_linked_opinion_wrappers(
            balance=self.__is_train(),
            supported_labels=self.__label_provider.SupportedLabels)

        for linked_wrap in linked_iter:

            for i in range(len(linked_wrap)):

                rows_it = self.__provide_rows(
                    opinion_provider=opinion_provider,
                    linked_wrap=linked_wrap,
                    index_in_linked=i,
                    idle_mode=idle_mode)

                for row in rows_it:
                    yield row

    # endregion

    def to_tsv_by_experiment(self, experiment, balance_rows_by_labels=True):
        assert(isinstance(experiment, BaseExperiment))

        filepath = self.get_filepath(data_type=self._data_type,
                                     experiment=experiment)

        if balance_rows_by_labels and self.__is_train():
            self._balance()

        self._df.to_csv(filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        float_format="%.0f",
                        compression='gzip',
                        header=not self.__is_train())

    def from_tsv(self, experiment):

        filepath = self.get_filepath(data_type=self._data_type,
                                     experiment=experiment)

        self._df = pd.read_csv(filepath,
                               compression='gzip',
                               sep='\t')

    def extract_ids(self):
        return self._df[self.ID].astype(unicode).tolist()

    @staticmethod
    def extract_row_id(opinion_row):
        assert(isinstance(opinion_row, list))
        return unicode(opinion_row[0])

    def __len__(self):
        return len(self._df.index)

    # private rows balancing functions

    @staticmethod
    def __get_class(df, label_provider, label):
        assert (isinstance(label, Label) or isinstance(label, int))

        if isinstance(label, Label):
            uint_label = label_provider.calculate_output_label_uint(
                expected_label=label,
                etalon_label=label)
        else:
            assert(label >= 0)
            uint_label = label

        return df[df[BaseSampleFormatter.LABEL] == uint_label]

    @staticmethod
    def __get_largest_class_size(df, label_provider, labels):
        assert(isinstance(label_provider, BertLabelProvider))

        sizes = [len(BaseSampleFormatter.__get_class(df=df, label_provider=label_provider, label=label))
                 for label in labels]

        return max(sizes)

    def _compose_balanced_df(self, largest_class_size, label, seed=1):
        """
        Composes a DataFrame which has the same amount of examples as one with 'other_label'
        """

        df = self._create_empty_df()
        self._fast_init_df(df=df, rows_count=largest_class_size)

        random.seed(seed)
        df_label = self.__get_class(df=self._df,
                                    label_provider=self.__label_provider,
                                    label=label)
        origin_size = len(df_label)

        if origin_size > 0:
            for row_index in xrange(len(df)):
                keep_row_index = row_index if row_index < origin_size else random.randint(0, origin_size - 1)
                row = df_label.iloc[keep_row_index]
                for column, value in row.iteritems():
                    self._set_value(df=df, row_ind=row_index, column=column, value=value)
        else:
            logger.info("Could not be expanded by label={}, as there are no related examples".format(label.to_class_str()))

        return df

    def _balance(self):
        """
        Balancing related dataframe by amount of examples per class
        """
        output_labels = self.__label_provider.OutputLabels

        largest_class_size = self.__get_largest_class_size(
            df=self._df,
            label_provider=self.__label_provider,
            labels=output_labels)

        balanced = [self._compose_balanced_df(label=label, largest_class_size=largest_class_size)
                    for label in output_labels]

        self._df = pd.concat(balanced)

        self._df.sort_values(by=[self.ID],
                             inplace=True,
                             ascending=True)

    # endregion
