import random
from collections import OrderedDict
from os import path

import numpy as np
import pandas as pd

import io_utils
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import Label
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.experiment.base import BaseExperiment
from arekit.contrib.bert.formatters.row_ids.binary import BinaryIDFormatter
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter
from arekit.contrib.bert.formatters.sample.label.base import LabelProvider
from arekit.contrib.bert.formatters.sample.label.binary import BinaryLabelProvider
from arekit.contrib.bert.formatters.sample.label.multiple import MultipleLabelProvider
from arekit.contrib.bert.formatters.sample.text.single import SingleTextProvider
from arekit.contrib.bert.formatters.utils import get_output_dir, generate_filename


class BaseSampleFormatter(object):
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
        assert(isinstance(text_provider, SingleTextProvider))

        self.__data_type = data_type
        # self.__id_col_index = self.__df.column.get_loc[BaseSampleFormatter.ID]
        self.__label_provider = label_provider
        self.__text_provider = text_provider
        self.__df = self.__create_empty_df()
        self.__row_ids_formatter = self.__create_row_ids_formatter(label_provider)

    # region Private methods

    @staticmethod
    def __create_row_ids_formatter(label_provider):
        if isinstance(label_provider, BinaryLabelProvider):
            return BinaryIDFormatter()
        if isinstance(label_provider, MultipleLabelProvider):
            return MultipleIDFormatter()

    def is_train(self):
        return self.__data_type == DataType.Train

    def __get_class(self, l):
        assert(isinstance(l, Label))
        return self.__df[self.__df[self.LABEL] == self.__label_provider.get_label(expected_label=l, etalon_label=l)]

    def __get_largest_class_size(self):

        sizes = [len(self.__get_class(label))
                 for label in self.__label_provider.get_supported_labels()]

        return max(sizes)

    def __extract_balanced_class(self, label, seed=1):
        """
        Composes a DataFrame which has the same amount of examples as one with 'other_label'
        """
        assert(isinstance(label, Label))

        df_label = self.__get_class(l=label)

        random.seed(seed)
        rows = []
        need_to_append = self.__get_largest_class_size() - len(df_label)

        while len(rows) < need_to_append:
            i = random.randint(0, len(df_label) - 1)
            rows.append(df_label.iloc[i])

        for row in rows:
            df_label = df_label.append(row)

        return df_label

    def __balance(self):
        """
        Balancing related dataframe by amount of examples per class
        """
        balanced = [self.__extract_balanced_class(label=label)
                    for label in self.__label_provider.get_supported_labels()]
        self.__df = pd.concat(balanced)

        self.__df = self.__df.iloc[np.random.permutation(len(self.__df))]

    # endregion

    def get_columns_list_with_types(self):
        """
        Composing df with the following columns:
            [id, label, type, text_a]
        """
        dtypes_list = []
        dtypes_list.append((self.ID, 'int32'))

        # insert labels
        if self.is_train():
            dtypes_list.append((self.LABEL, 'int32'))

        # insert text columns
        for col_name in self.__text_provider.iter_columns():
            dtypes_list.append((col_name, 'float64'))

        # insert indices
        dtypes_list.append((self.S_IND, 'int32'))
        dtypes_list.append((self.T_IND, 'int32'))

        return dtypes_list

    def __create_empty_df(self):
        data = np.empty(0, dtype=np.dtype(self.get_columns_list_with_types()))
        return pd.DataFrame(data)

    @staticmethod
    def __get_opinion_end_indices(parsed_news, text_opinion):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.SourceId)
        t_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.TargetId)

        return (s_ind, t_ind)

    def __create_row(self, parsed_news, linked_text_opinions, index_in_linked, sentence_terms, etalon_label):
        """
        Composing row in following format:
            [id, label, type, text_a]

        returns: OrderedDict
            row with key values
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(sentence_terms, list))
        assert(isinstance(etalon_label, Label))

        first_text_opinion = linked_text_opinions[0]
        text_opinion = linked_text_opinions[index_in_linked]

        assert(isinstance(first_text_opinion, TextOpinion))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind, t_ind = self.__get_opinion_end_indices(parsed_news, text_opinion)

        row = OrderedDict()

        row[self.ID] = self.__row_ids_formatter.create_sample_id(
            first_text_opinion=first_text_opinion,
            index_in_linked=index_in_linked)

        if self.is_train():
            row[self.LABEL] = self.__label_provider.get_label(expected_label=text_opinion.Sentiment,
                                                              etalon_label=etalon_label)

        self.__text_provider.add_text_in_row(row=row,
                                             sentence_terms=sentence_terms,
                                             s_ind=s_ind,
                                             t_ind=t_ind)

        row[self.S_IND] = s_ind
        row[self.T_IND] = t_ind

        return row

    def __provide_rows(self, parsed_news, linked_text_opinions, index_in_linked, sentence_terms):
        """
        Providing Rows depending on row_id_formatter type
        """

        origin = linked_text_opinions[0]

        if isinstance(self.__row_ids_formatter, BinaryIDFormatter):
            """
            Enumerate all opinions as if it would be with the different label types.
            """
            for label in Label._get_supported_labels():
                copy = TextOpinion.create_copy(other=origin)
                copy.set_label(label=label)
                linked_text_opinions[0] = copy
                yield self.__create_row(parsed_news=parsed_news,
                                        linked_text_opinions=linked_text_opinions,
                                        index_in_linked=index_in_linked,
                                        sentence_terms=sentence_terms,
                                        etalon_label=origin.Sentiment)

        if isinstance(self.__row_ids_formatter, MultipleIDFormatter):
            yield self.__create_row(parsed_news=parsed_news,
                                    linked_text_opinions=linked_text_opinions,
                                    index_in_linked=index_in_linked,
                                    sentence_terms=sentence_terms,
                                    etalon_label=origin.Sentiment)

    def to_samples(self, text_opinions):
        """
        Converts text_opinions into samples by filling related df.
        """
        assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))

        pnc = text_opinions.RelatedParsedNewsCollection
        assert(isinstance(pnc, ParsedNewsCollection))

        added = 0

        for linked_opinions in text_opinions.iter_by_linked_text_opinions():

            for i, text_opinion in enumerate(linked_opinions):
                assert(isinstance(text_opinion, TextOpinion))

                # Determining text_a by text_opinion end.
                s_ind = TextOpinionHelper.extract_entity_sentence_index(
                    text_opinion=text_opinion,
                    end_type=EntityEndType.Source)

                # Extract specific document by text_opinion.NewsID
                pn = pnc.get_by_news_id(text_opinion.NewsID)

                assert(isinstance(pn, ParsedNews))

                # Extract text_a terms.
                sentence_terms = list(pn.iter_sentence_terms(s_ind))

                # Add text_an
                rows_it = self.__provide_rows(linked_text_opinions=linked_opinions,
                                              index_in_linked=i,
                                              sentence_terms=sentence_terms,
                                              parsed_news=pn)

                for row in rows_it:
                    self.__df = self.__df.append(row, ignore_index=True)

                added += 1

            print "Samples ('{}') added: {}/{} ({}%)".format(
                self.__data_type,
                added,
                len(text_opinions),
                round(100 * float(added) / len(text_opinions), 2))

        if self.is_train() and isinstance(self.__row_ids_formatter, MultipleLabelProvider):
            self.__balance()

    def to_tsv_by_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))

        filepath = self.get_filepath(data_type=self.__data_type,
                                     experiment=experiment)

        self.__df.to_csv(filepath,
                         sep='\t',
                         encoding='utf-8',
                         index=False,
                         header=not self.is_train())

    @staticmethod
    def extract_row_id(opinion_row):
        assert(isinstance(opinion_row, list))
        return unicode(opinion_row[0])

    @staticmethod
    def get_filepath(data_type, experiment):
        assert(isinstance(experiment, BaseExperiment))

        fname = generate_filename(data_type=data_type,
                                  experiment=experiment,
                                  prefix=u'samples')

        filepath = path.join(get_output_dir(experiment=experiment), fname)

        io_utils.create_dir_if_not_exists(filepath)

        return filepath

