from arekit.common.labels.base import Label


class TextOpinion(object):
    """
    Represents a relation which were found in news article
    and composed between two named entities
        (it was found especially by Opinion with predefined label)
        allows to modify label using set_label
    """

    # region constructors

    def __init__(self, news_id, text_opinion_id, source_id, target_id, owner, label):
        assert(isinstance(news_id, int))
        assert(isinstance(text_opinion_id, int) or text_opinion_id is None)

        self.__source_id = source_id
        self.__target_id = target_id
        self.__news_id = news_id
        self.__owner = owner
        self.__text_opinion_id = text_opinion_id
        self.__modifiable_label = label

    @classmethod
    def create_copy(cls, other):
        assert(isinstance(other, TextOpinion))
        return TextOpinion(news_id=other.__news_id,
                           text_opinion_id=other.__text_opinion_id,
                           source_id=other.SourceId,
                           target_id=other.TargetId,
                           owner=other.Owner,
                           label=other.Sentiment)

    # endregion

    # region properties

    @property
    def Sentiment(self):
        return self.__modifiable_label

    @property
    def NewsID(self):
        return self.__news_id

    @property
    def TextOpinionID(self):
        return self.__text_opinion_id

    @property
    def SourceId(self):
        return self.__source_id

    @property
    def TargetId(self):
        return self.__target_id

    @property
    def Owner(self):
        return self.__owner

    # endregion

    # region public methods

    def set_text_opinion_id(self, text_opinion_id):
        assert(self.__text_opinion_id is None)
        assert(isinstance(text_opinion_id, int))
        self.__text_opinion_id = text_opinion_id

    def set_label(self, label):
        assert(isinstance(label, Label))
        self.__modifiable_label = label

    def set_owner(self, owner):
        assert(owner is not None)
        assert(self.__owner is None)
        self.__owner = owner

    # endregion
