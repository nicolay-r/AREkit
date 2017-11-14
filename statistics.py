#!/usr/bin/python

import pandas as pd

from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection

import io_utils

root = io_utils.train_root()

columns = ('entities', 'sentences', 's_opins', 'n_opins', 'syn_groups')
df = pd.DataFrame(columns=columns)

for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)
    news_filepath = root + "art{}.txt".format(n)

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)
    sentiment_opins = OpinionCollection.from_file(
        opin_filepath, io_utils.get_synonyms_filepath())
    neutral_opins = OpinionCollection.from_file(
        neutral_filepath, io_utils.get_synonyms_filepath())

    row = [entities.count(),
           len(news.sentences),
           len(sentiment_opins),
           len(neutral_opins),
           sentiment_opins.synonyms._get_groups_count()]

    df.loc[n] = row

df.loc['avg'] = [float(df[['{}'.format(c)]].mean()) for c in columns]

df.to_csv("statistics.txt")
