from collections import OrderedDict

import numpy as np
import pandas as pd

import io_utils
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.contrib.bert_encoder.io_utils import get_experiments_dir


# region private functions


def __create_empty_df():
    dtypes_list = []
    dtypes_list.append(('id', 'int32'))
    dtypes_list.append(('source', 'string'))
    dtypes_list.append(('target', 'string'))

    data = np.empty(0, dtype=np.dtype(dtypes_list))
    return pd.DataFrame(data)


def __create_opinion_row(linked_text_opinion):
    """
    row format: [id, src, target, label]
    """

    row = OrderedDict()

    src_value = TextOpinionHelper.extract_entity_value(
        text_opinion=linked_text_opinion,
        end_type=EntityEndType.Source)

    target_value = TextOpinionHelper.extract_entity_value(
        text_opinion=linked_text_opinion,
        end_type=EntityEndType.Target)

    row['id'] = create_opinion_id(first_text_opinion=linked_text_opinion,
                                  index_in_linked=0)
    row['source'] = src_value
    row['target'] = target_value

    return row

# endregion

def parse_row(df_row):
    assert(isinstance(df_row, list))

    news_id = df_row[0]
    source = df_row[1].decode('utf-8')
    target = df_row[2].decode('utf-8')

    return news_id, source, target


def create_opinion_id(first_text_opinion, index_in_linked):
    assert(isinstance(first_text_opinion, TextOpinion))
    assert(isinstance(index_in_linked, int))

    return u"n{}_o{}_i{}".format(first_text_opinion.NewsID,
                                 first_text_opinion.TextOpinionID,
                                 index_in_linked)


def create_and_save_opinions_to_csv(text_opinions, data_type, model_name):
    assert(isinstance(model_name, unicode))
    assert(isinstance(data_type, unicode))

    df = __create_empty_df()

    print "Adding opinions ('{}') ... ".format(data_type)

    added = 0
    for linked_opinions in text_opinions.iter_by_linked_text_opinions():

        added += 1

        row = __create_opinion_row(linked_text_opinion=linked_opinions[0])

        df = df.append(row, ignore_index=True)

    df.to_csv(get_filepath(data_type=data_type, model_name=model_name),
              sep='\t',
              encoding='utf-8',
              index=False,
              header=False)


def sample_row_id_to_opinion_id(row_id):
    """
    Id in sample rows has information of linked opinions.
    Here the latter ommited and id could be suffexed with 'i0' only.
    """
    assert(isinstance(row_id, unicode))
    return row_id[:row_id.find(u'i')] + u"i0"


def get_filepath(data_type, model_name):
    assert(isinstance(model_name, unicode))
    assert(isinstance(data_type, unicode))

    filepath = u"{dir}{model_name}/{filename}.csv".format(
        dir=get_experiments_dir(),
        model_name=model_name,
        filename=u"{}-opinions".format(data_type))

    io_utils.create_dir_if_not_exists(filepath)

    return filepath

