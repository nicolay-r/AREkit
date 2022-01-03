from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.base import BaseParsedNewsServiceProvider


class ParsedNewsService(object):
    """ Represents a collection of providers, combined with the parsed news.
    """

    def __init__(self, parsed_news, providers):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(providers, list))
        self.__parsed_news = parsed_news
        self.__providers = {}

        for provider in providers:
            assert(isinstance(provider, BaseParsedNewsServiceProvider))
            assert(provider.Name not in self.__providers)
            self.__providers[provider.Name] = provider

    @property
    def ParsedNews(self):
        return self.__parsed_news

    def get_provider(self, name):
        return self.__providers[name]