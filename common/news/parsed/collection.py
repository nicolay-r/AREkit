import collections
import logging

from arekit.common.news.parsed.base import ParsedNews
from arekit.common.utils import progress_bar_iter


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
        for parsed_news in progress_bar_iter(parsed_news_it, desc="Filling parsed news"):
            assert(isinstance(parsed_news, ParsedNews))
            if parsed_news in d:
                logging.info("Warning: Skipping document with id={}".format(parsed_news.RelatedNewsID))
                continue

            d[parsed_news.RelatedNewsID] = parsed_news

        return d

    def get_by_news_id(self, news_id):
        assert(isinstance(news_id, int))
        return self.__by_id[news_id]

    def iter_news_terms(self, news_id, term_check=None):
        assert(isinstance(news_id, int))
        for term in self.__by_id[news_id].iter_terms(term_check=term_check):
            yield term

    def iter_news_ids(self):
        for news_id in self.__by_id.iterkeys():
            yield news_id

    def __contains__(self, item):
        assert(isinstance(item, int))
        return item in self.__by_id
