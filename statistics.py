#!/usr/bin/python

import pandas as pd

from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection

import io_utils


def neutral_in_senitmen_opins(n_opins, s_opins):
    assert(isinstance(n_opins, OpinionCollection))
    assert(isinstance(s_opins, OpinionCollection))
    founded = 0
    for e in n_opins:
        founded += 1 if s_opins.has_opinion_by_synonyms(e) else 0
    return founded

def expand(df):
    df.loc['avg'] = [float(df[['{}'.format(c)]].mean()) for c in columns]
    # df.loc['ans_recall'] = float(float(df.loc['avg']['neut_in_sent']) / df.loc['avg']['s_opins'])


def calc_file_info(news_filepath, entities_filepath, opin_filepath, neut_filepath):
    entities = EntityCollection.from_file(entities_filepath)
    news = News.from_file(news_filepath, entities)

    s_opins = OpinionCollection.from_file(
        opin_filepath, io_utils.get_synonyms_filepath())
    n_opins = OpinionCollection.from_file(
        neut_filepath, io_utils.get_synonyms_filepath())

    row = [entities.count(),
           len(news.sentences),
           len(s_opins),
           len(n_opins),
           s_opins.synonyms._get_groups_count(),
           neutral_in_senitmen_opins(n_opins, s_opins)]

    return row


columns = ('entities', 'sentences', 's_opins', 'n_opins', 'syn_groups', 'neut_in_sent')

df_train = pd.DataFrame(columns=columns)
for n in io_utils.train_indices():
    root = io_utils.train_root()

    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neut_filepath = root + "art{}.neut.txt".format(n)

    df_train.loc[n] = calc_file_info(
        news_filepath, entity_filepath, opin_filepath, neut_filepath)

df_test = pd.DataFrame(columns=columns)
for n in io_utils.test_indices():
    root = io_utils.test_root()

    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    opin_filepath = io_utils.get_etalon_root() + "art{}.opin.txt".format(n)
    neut_filepath = root + "art{}.neut.txt".format(n)

    df_test.loc[n] = calc_file_info(
        news_filepath, entity_filepath, opin_filepath, neut_filepath)

expand(df_train)
expand(df_test)

df_train.to_csv(io_utils.data_root() + "train_statistics.txt")
df_test.to_csv(io_utils.data_root() + "test_statistics.txt")
