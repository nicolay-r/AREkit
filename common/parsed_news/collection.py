# -*- coding: utf-8 -*-
import collections
import logging

from arekit.common.parsed_news.base import ParsedNews


class ParsedNewsCollection(object):
    """
    This collection stores processed news (parsed),
    which could be indentified by news_id.
    """

    def __init__(self, parsed_news_it):
        assert(isinstance(parsed_news_it, collections.Iterable))
        self.__by_id = self.__fill(parsed_news_it)

    @staticmethod
    def __fill(parsed_news_it):
        assert(isinstance(parsed_news_it, collections.Iterable))

        d = {}
        for parsed_news in parsed_news_it:
            assert(isinstance(parsed_news, ParsedNews))
            if parsed_news in d:
                logging.info("Warning: Skipping document with id={}".format(parsed_news.RelatedNewsID))
                continue

            d[parsed_news.RelatedNewsID] = parsed_news

        return d

    def get_by_news_id(self, news_id):
        assert(isinstance(news_id, int))
        return self.__by_id[news_id]

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
