from core.evaluation.labels import Label


class RefOpinion(object):
    """
    Provides references within Owner collection.
    """

    def __init__(self, left_index, right_index, sentiment, owner=None):
        assert(isinstance(left_index, int))
        assert(isinstance(right_index, int))
        assert(isinstance(sentiment, Label))
        self.__left_index = left_index
        self.__rigth_index = right_index
        self.__sentiment = sentiment
        self.__owner = owner

    @property
    def LeftIndex(self):
        return self.__left_index

    @property
    def RightIndex(self):
        return self.__rigth_index

    @property
    def Sentiment(self):
        return self.__sentiment

    @property
    def Tag(self):
        return self.__tag

    def set_tag(self, value):
        self.__tag = value
