from arekit.common.text_opinions.base import TextOpinion


class TextOpinionCollection(object):
    """
    Collection of text-level opinions across many news/documents
    """

    def __init__(self, text_opinions):
        """
        parsed_news_collection: ParsedNewsCollection
            utilized as reference only (for Helper)
        text_opinions: list
            list of TextOpinion
        """
        assert(isinstance(text_opinions, list))

        self.__text_opinions = text_opinions

    # region protected methods

    def _remove_last_registered_text_opinion(self):
        del self.__text_opinions[-1]

    # endregion

    # region public methods

    def register_text_opinion(self, text_opinion):
        assert(isinstance(text_opinion, TextOpinion))
        self.__text_opinions.append(text_opinion)

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
