from arekit.common.text_opinions.base import TextOpinion
from arekit.common.parsed_news.collection import ParsedNewsCollection


class TextOpinionCollection(object):
    """
    Collection of text-level opinions

    Limitations:
        IN MEMORY implemenatation.
    """

    def __init__(self, parsed_news_collection, text_opinions):
        assert(isinstance(parsed_news_collection, ParsedNewsCollection) or
               parsed_news_collection is None)
        assert(isinstance(text_opinions, list))

        self.__parsed_news_collection = parsed_news_collection
        self.__text_opinions = text_opinions

    # region property

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

    def iter_unique_news_ids(self):
        unique_news_ids = set()

        for text_opinion in self.__text_opinions:
            assert(isinstance(text_opinion, TextOpinion))
            id = text_opinion.NewsID
            if id not in unique_news_ids:
                unique_news_ids.add(id)
                yield id

    def iter_text_opinons_in_doc_ids_set(self, doc_ids_set):
        assert(isinstance(doc_ids_set, set))
        for text_opinion in self.__text_opinions:
            if text_opinion.NewsID in doc_ids_set:
                yield text_opinion

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
