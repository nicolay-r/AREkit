#!/usr/bin/python
from pexpect import searcher_re

import pandas as pd

from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection

import io_utils


def founded_opins(test_opins, etalon_opins, sentiment=None):
    assert(isinstance(test_opins, OpinionCollection))
    assert(isinstance(etalon_opins, OpinionCollection))
    founded = 0
    for e in test_opins:
        founded += 1 if etalon_opins.has_opinion_by_synonyms(e, sentiment) else 0
    return founded


def get_method_statistic(method_name):
    columns = ["t_all", # "t_pos", "t_neg",
               "e_all", # "e_pos", "e_neg"
               ]
    df = pd.DataFrame(columns=columns)
    for n in io_utils.test_indices():
        root = io_utils.test_root() + '{}/'.format(method_name)

        eo_filepath = io_utils.get_etalon_root() + "art{}.opin.txt".format(n)
        to_filepath = root + "art{}.opin.txt".format(n)

        test_opins = OpinionCollection.from_file(to_filepath, io_utils.get_synonyms_filepath())
        etalon_opins = OpinionCollection.from_file(eo_filepath, io_utils.get_synonyms_filepath())

        df.loc[n] = [founded_opins(test_opins, etalon_opins),
                     # founded_opins(test_opins, etalon_opins, u'pos'),
                     # founded_opins(test_opins, etalon_opins, u'neg'),
                     len(etalon_opins),
                     # len(list(etalon_opins.iter_sentiment(u'pos'))),
                     # len(list(etalon_opins.iter_sentiment(u'neg')))
                     ]

    df.loc['sum'] = [float(df[c].sum()) for c in columns]

    df.loc['founded_all'] = None
    # df.loc['founded_pos'] = None
    # df.loc['founded_neg'] = None
    df.loc['founded_all'][0] = float(df.loc['sum']['t_all']) / df.loc['sum']['e_all']
    # df.loc['founded_pos'][1] = float(df.loc['sum']['t_pos']) / df.loc['sum']['e_pos']
    # df.loc['founded_neg'][2] = float(df.loc['sum']['t_neg']) / df.loc['sum']['e_neg']
    return df


def expand(df):
    founded = float(df['neut_in_sent'].sum()) / df['s_opins'].sum()
    df.loc['avg'] = [float(df[c].mean()) for c in columns]
    df.loc['founded'] = None
    df.loc['founded'][0] = founded


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
           founded_opins(n_opins, s_opins)]

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

methods = ['svm', 'knn', 'rf', 'nb']

for m in methods:
    df = get_method_statistic(m)
    df.to_csv(io_utils.data_root() + 'test_{}.csv'.format(m))
