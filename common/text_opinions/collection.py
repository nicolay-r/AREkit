from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.parsed_news.collection import ParsedNewsCollection


class TextOpinionCollection(object):
    """
    Collection of text-level opinions across many news/documents
    """

    # TODO. Parsed news collection (remove)
    def __init__(self, parsed_news_collection, text_opinions):
        """
        parsed_news_collection: ParsedNewsCollection
            utilized as reference only (for Helper)
        text_opinions: list
            list of TextOpinion
        """
        assert(isinstance(parsed_news_collection, ParsedNewsCollection) or
               parsed_news_collection is None)
        assert(isinstance(text_opinions, list))

        # TODO. This should be removed
        self.__parsed_news_collection = parsed_news_collection
        self.__text_opinions = text_opinions

    # region property

    # TODO. This should be removed
    # TODO. This should be a part of Dataset class
    @property
    def RelatedParsedNewsCollection(self):
        return self.__parsed_news_collection

    # endregion

    # region public methods

    def register_text_opinion(self, text_opinion):
        assert(isinstance(text_opinion, TextOpinion))
        self.__text_opinions.append(text_opinion)

    def remove_last_registered_text_opinion(self):
        del self.__text_opinions[-1]

    def get_unique_news_ids(self):
        return set(map(lambda text_opinion: text_opinion.NewsID, self))

    # endregion

    # region base methods

    def __iter__(self):
        for text_opinion in self.__text_opinions:
            yield text_opinion

    def __len__(self):
        return len(self.__text_opinions)

    def __getitem__(self, item):
        assert(isinstance(item,  int))
        return self.__text_opinions[item]

    # endregion
