import random
from collections import OrderedDict
from os import path

import numpy as np
import pandas as pd

import io_utils
from arekit.common.experiment.data_type import DataType
from arekit.common.entities.base import Entity
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.bert.formatters.opinion import OpinionsFormatter
from arekit.common.experiment.base import BaseExperiment
from arekit.contrib.bert.formatters.utils import get_output_dir, generate_filename
from arekit.processing.text.token import Token


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
    TEXT_A = 'text_a'
    S_IND = 's_ind'
    T_IND = 't_ind'

    SUBJECT = u"X"
    OBJECT = u"Y"

    TERMS_SEPARATOR = u" "

    def __init__(self, data_type):
        self.__data_type = data_type
        self.__df = self.__create_empty_df()

    # region Private methods

    def is_train(self):
        return self.__data_type == DataType.Train

    @staticmethod
    def __iterate_sentence_terms(sentence_terms, s_ind, t_ind):

        for i, term in enumerate(sentence_terms):

            if isinstance(term, unicode):
                yield term
            elif isinstance(term, Entity):
                if i == s_ind:
                    yield BaseSampleFormatter.SUBJECT
                elif i == t_ind:
                    yield BaseSampleFormatter.OBJECT
                else:
                    u"E"
            elif isinstance(term, Token):
                yield term.get_original_value()

    def __balance(self, label, other_label=0, seed=1):
        """
        Composes a DataFrame which has the same amount of examples as one with 'other_label'
        """
        assert(isinstance(label, int))
        assert(isinstance(other_label, int))
        df_label = self.__df[self.__df[self.LABEL] == label]
        df_other_label = self.__df[self.__df[self.LABEL] == other_label]

        random.seed(seed)
        rows = []
        need_to_append = len(df_other_label) - len(df_label)

        while len(rows) < need_to_append:
            i = random.randint(0, len(df_label) - 1)
            rows.append(df_label.iloc[i])

        for row in rows:
            df_label = df_label.append(row)

        return df_label

    # endregion

    def get_columns_list_with_types(self):
        """
        Composing df with the following columns:
            [id, label, type, text_a]
        """
        dtypes_list = []
        dtypes_list.append((self.ID, 'int32'))

        if self.is_train():
            dtypes_list.append((self.LABEL, 'int32'))

        dtypes_list.append((self.TEXT_A, 'float64'))
        dtypes_list.append((self.S_IND, 'int32'))
        dtypes_list.append((self.T_IND, 'int32'))

        return dtypes_list

    def __create_empty_df(self):
        data = np.empty(0, dtype=np.dtype(self.get_columns_list_with_types()))
        return pd.DataFrame(data)

    @staticmethod
    def __get_opinion_end_inices(parsed_news, text_opinion):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.SourceId)
        t_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.TargetId)

        return (s_ind, t_ind)

    def create_row(self, parsed_news, linked_text_opinions, index_in_linked, sentence_terms):
        """
        Composing row in following format:
            [id, label, type, text_a]

        returns: OrderedDict
            row with key values
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(sentence_terms, list))

        first_text_opinion = linked_text_opinions[0]
        text_opinion = linked_text_opinions[index_in_linked]

        assert(isinstance(first_text_opinion, TextOpinion))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind, t_ind = self.__get_opinion_end_inices(parsed_news, text_opinion)

        row = OrderedDict()

        row[self.ID] = OpinionsFormatter.create_opinion_id(first_text_opinion=first_text_opinion,
                                                           index_in_linked=index_in_linked)

        if self.is_train():
            row[self.LABEL] = text_opinion.Sentiment.to_uint()

        row[self.TEXT_A] = self.TERMS_SEPARATOR.join(self.__iterate_sentence_terms(sentence_terms, s_ind=s_ind, t_ind=t_ind))
        row[self.S_IND] = s_ind
        row[self.T_IND] = t_ind

        return row

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
                row = self.create_row(linked_text_opinions=linked_opinions,
                                      index_in_linked=i,
                                      sentence_terms=sentence_terms,
                                      parsed_news=pn)

                self.__df = self.__df.append(row, ignore_index=True)

                added += 1

            print "Samples ('{}') added: {}/{} ({}%)".format(
                self.__data_type,
                added,
                len(text_opinions),
                round(100 * float(added) / len(text_opinions), 2))

        if self.is_train():

            df_neut = self.__df[self.__df[self.LABEL] == 0]
            df_pos = self.__balance(label=1)
            df_neg = self.__balance(label=2)

            self.__df = pd.concat([df_neut, df_neg, df_pos])

            self.__df = self.__df.iloc[np.random.permutation(len(self.__df))]

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
    def parse_row_id(opinion_row):
        assert(isinstance(opinion_row, list))
        return unicode(opinion_row[0])

    @staticmethod
    def parse_news_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(u'n') + 1:row_id.index(u'_')])

    @staticmethod
    def get_filepath(data_type, experiment):
        assert(isinstance(experiment, BaseExperiment))

        fname = generate_filename(data_type=data_type,
                                  experiment=experiment,
                                  prefix=u'samples')

        filepath = path.join(get_output_dir(experiment=experiment), fname)

        io_utils.create_dir_if_not_exists(filepath)

        return filepath

