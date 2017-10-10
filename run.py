#!/usr/bin/python
# -*- coding: utf-8 -*-

from core.annot import EntityCollection
from core.news import News

annot_filepath = "data/Texts/art2.ann"
news_filepath = 'data/Texts/art2.txt'

entities = EntityCollection.from_file(annot_filepath)
news = News.from_file(news_filepath, entities)
news.show()
