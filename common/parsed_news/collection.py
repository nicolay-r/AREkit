# -*- coding: utf-8 -*-
from arekit.common.parsed_news.base import ParsedNews


class ParsedNewsCollection(object):
    """
    This collection stores processed news (parsed),
    which could be indentified by news_id.
    """

    def __init__(self):
        self.__by_id = {}

    def get_by_news_id(self, news_id):
        assert(isinstance(news_id, int))
        return self.__by_id[news_id]

    # TODO. IN memory implementation.
    # TODO. Implement a different class which support external storages.
    # TODO. Remove this method (make init from iter as set)
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

    def __contains__(self, item):
        assert(isinstance(item, int))
        return item in self.__by_id
