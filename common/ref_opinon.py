from core.common.labels.base import Label


class RefOpinion(object):
    """
    Provides references within Owner collection with id's.
    """

    def __init__(self, source_id, target_id, sentiment, owner=None):
        assert(isinstance(source_id, int))
        assert(isinstance(target_id, int))
        assert(isinstance(sentiment, Label))
        self.__source_id = source_id
        self.__target_id = target_id
        self.__sentiment = sentiment
        self.__owner = owner
        self.__tag = None

    # region properties

    @property
    def SourceId(self):
        return self.__source_id

    @property
    def TargetId(self):
        return self.__target_id

    @property
    def Owner(self):
        return self.__owner

    @property
    def Sentiment(self):
        return self.__sentiment

    @property
    def Tag(self):
        return self.__tag

    # endregion

    # region public methods

    def set_tag(self, value):
        self.__tag = value

    def set_owner(self, owner):
        assert(owner is not None)
        assert(self.__owner is None)
        self.__owner = owner

    # endregion
