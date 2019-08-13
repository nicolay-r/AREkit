from collections import OrderedDict
from core.common.text_opinions.text_opinion import TextOpinion
from core.common.parsed_news.collection import ParsedNewsCollection


class TextOpinionCollection(object):

    def __init__(self, parsed_news_collection, text_opinions):
        assert(isinstance(parsed_news_collection, ParsedNewsCollection) or
               parsed_news_collection is None)
        assert(isinstance(text_opinions, list))

        self.__parsed_news_collection = parsed_news_collection
        self.__text_opinions = text_opinions

    @property
    def RelatedParsedNewsCollection(self):
        return self.__parsed_news_collection

    def register_text_opinion(self, text_opinion):
        assert(isinstance(text_opinion, TextOpinion))
        self.__text_opinions.append(text_opinion)

    def iter_unique_news_ids(self):
        unique_news_ids = set()

        for text_opinion in self.__text_opinions:
            assert(isinstance(text_opinion, TextOpinion))
            id = text_opinion.NewsID
            if id not in unique_news_ids:
                unique_news_ids.add(id)
                yield id

    def __iter__(self):
        for text_opinion in self.__text_opinions:
            yield text_opinion

    def __len__(self):
        return len(self.__text_opinions)

    def __getitem__(self, item):
        assert(isinstance(item,  int))
        return self.__text_opinions[item]
