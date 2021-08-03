from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion


class SentenceOpinion(object):
    """
    Provides an opinion within a sentence.
    Specific for RuAttitudes collection, as the latter provides connections within a sentence.
    """

    def __init__(self, source_id, target_id, source_value, target_value, sentiment, tag):
        assert(isinstance(source_id, int))
        assert(isinstance(target_id, int))
        assert(isinstance(source_value, unicode))
        assert(isinstance(target_value, unicode))
        assert(isinstance(sentiment, Label))

        self.__source_id = source_id
        self.__target_id = target_id
        self.__source_value = source_value
        self.__target_value = target_value
        self.__sentiment = sentiment
        self.__tag = tag

    # region properties

    @property
    def SourceID(self):
        return self.__source_id

    @property
    def TargetID(self):
        return self.__target_id

    @property
    def Sentiment(self):
        return self.__sentiment

    @property
    def Tag(self):
        return self.__tag

    # endregion

    def to_text_opinion(self, news_id, end_to_doc_id_func, text_opinion_id):
        """
        Converts opinion into document-level referenced opinion
        """
        return TextOpinion(news_id=news_id,
                           text_opinion_id=text_opinion_id,
                           source_id=end_to_doc_id_func(self.__source_id),
                           target_id=end_to_doc_id_func(self.__target_id),
                           owner=None,
                           label=self.__sentiment)

    def to_opinion(self):
        """
        Converts onto document, non referenced opinion
        (non bounded to the text).
        """
        opinion = Opinion(source_value=self.__source_value,
                          target_value=self.__target_value,
                          sentiment=self.__sentiment)

        # Using this tag allows to perform a revert operation,
        # i.e. to find opinion_ref by opinion.
        opinion.set_tag(self.__tag)

        return opinion
