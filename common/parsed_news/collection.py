# -*- coding: utf-8 -*-
from core.common.parsed_news.parsed_news import ParsedNews


class ParsedNewsCollection(object):

    def __init__(self):
        self.__by_id = {}

    def get_by_news_id(self, news_id):
        assert(isinstance(news_id, int))
        return self.__by_id[news_id]

    def add(self, parsed_news):
        assert(isinstance(parsed_news, ParsedNews))
        assert(parsed_news.RelatedNewsID not in self.__by_id)
        self.__by_id[parsed_news.RelatedNewsID] = parsed_news

    def iter_news_terms(self, news_id):
        assert(isinstance(news_id, int))
        for term in self.__by_id[news_id].iter_terms():
            yield term

    def iter_news_ids(self):
        for news_id in self.__by_id.iterkeys():
            yield news_id
