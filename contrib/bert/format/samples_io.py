import random
from collections import OrderedDict

import numpy as np
import pandas as pd

import io_utils
from arekit.common.entities.base import Entity
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper

from arekit.networks.data_type import DataType
from arekit.processing.text.token import Token
from opinions_io import create_opinion_id


# region private functions

def __create_empty_df(data_type):
    assert(isinstance(data_type, unicode))

    dtypes_list = []
    dtypes_list.append(('id', 'int32'))

    if data_type == DataType.Train:
        dtypes_list.append(('label', 'int32'))
        dtypes_list.append(('type', 'string'))

    dtypes_list.append(('sentence', 'float64'))

    data = np.empty(0, dtype=np.dtype(dtypes_list))
    return pd.DataFrame(data)


def __create_row(parsed_news,
                 linked_text_opinions,
                 index_in_linked,
                 sentence_terms,
                 data_type):
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(sentence_terms, list))
    assert(isinstance(data_type, unicode))

    first_text_opinion = linked_text_opinions[0]
    text_opinion = linked_text_opinions[index_in_linked]

    assert(isinstance(first_text_opinion, TextOpinion))
    assert(isinstance(text_opinion, TextOpinion))

    s_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.SourceId)
    t_ind = parsed_news.get_entity_sentence_level_term_index(text_opinion.TargetId)

    row = OrderedDict()

    row['id'] = create_opinion_id(first_text_opinion=first_text_opinion,
                                  index_in_linked=index_in_linked)

    if data_type == DataType.Train:
        row['label'] = text_opinion.Sentiment.to_uint()
        row['type'] = 'a'

    row['sentence'] = u" ".join(__iterate_sentence_terms(sentence_terms, s_ind=s_ind, t_ind=t_ind))

    return row


def __iterate_sentence_terms(sentence_terms, s_ind, t_ind):

    for i, term in enumerate(sentence_terms):

        if isinstance(term, unicode):
            yield term
        elif isinstance(term, Entity):
            if i == s_ind:
                yield u"$ E $"
            elif i == t_ind:
                yield u"# E #"
            else:
                u"E"
        elif isinstance(term, Token):
            yield term.get_original_value()


def balance(df, label, other_label=0, seed=1):
    """
    Composes a DataFrame which has the same amount of examples as one with 'other_label'
    """
    assert(isinstance(df, pd.DataFrame))
    assert(isinstance(label, int))
    assert(isinstance(other_label, int))
    df_label = df[df['label'] == label]
    df_other_label = df[df['label'] == other_label]

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


def create_and_save_samples_to_tsv(text_opinions, pnc, data_type, model_name):
    """
        Train/Test input samples for BERT
    """
    assert(isinstance(data_type, unicode))
    assert(isinstance(model_name, unicode))
    assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))
    assert(isinstance(pnc, ParsedNewsCollection))

    df = __create_empty_df(data_type)

    added = 0

    for linked_opinions in text_opinions.iter_by_linked_text_opinions():

        for i, text_opinion in enumerate(linked_opinions):
            assert(isinstance(text_opinion, TextOpinion))

            # Determining sentence by text_opinion end.
            s_ind = TextOpinionHelper.extract_entity_sentence_index(
                text_opinion=text_opinion,
                end_type=EntityEndType.Source)

            # Extract specific document by text_opinion.NewsID
            pn = pnc.get_by_news_id(text_opinion.NewsID)

            assert(isinstance(pn, ParsedNews))

            # Extract sentence terms.
            sentence_terms = list(pn.iter_sentence_terms(s_ind))

            # Add sentence
            row = __create_row(linked_text_opinions=linked_opinions,
                               index_in_linked=i,
                               sentence_terms=sentence_terms,
                               parsed_news=pn,
                               data_type=data_type)

            df = df.append(row, ignore_index=True)

            added += 1

        print "Samples ('{}') added: {}/{} ({}%)".format(
            data_type,
            added,
            len(text_opinions),
            round(100 * float(added) / len(text_opinions), 2))

    if data_type == DataType.Train:

        df_neut = df[df['label'] == 0]
        df_pos = balance(df=df, label=1)
        df_neg = balance(df=df, label=2)

        df = pd.concat([df_neut, df_neg, df_pos])

        df = df.iloc[np.random.permutation(len(df))]

    df.to_csv(get_filepath(model_name=model_name, data_type=data_type),
              sep='\t',
              encoding='utf-8',
              index=False,
              header=data_type == DataType.Test)


def parse_row_id(opinion_row):
    assert(isinstance(opinion_row, list))
    return unicode(opinion_row[0])


def parse_news_id(row_id):
    assert(isinstance(row_id, unicode))
    return int(row_id[row_id.index(u'n') + 1:row_id.index(u'_')])


def get_filepath(model_name, data_type):
    assert(isinstance(model_name, unicode))
    assert(isinstance(data_type, unicode))

    filepath = u"{dir}{model_name}/{filename}.tsv".format(
        dir=io_utils.get_experiments_dir(),
        model_name=model_name,
        filename=data_type)

    io_utils.create_dir_if_not_exists(filepath)

    return filepath

