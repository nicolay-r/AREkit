class RefOpinion(object):
    """
    Provides references within Owner collection.
    """

    def __init__(self, left_index, right_index, owner):
        assert(isinstance(left_index, int))
        assert(isinstance(right_index, int))
        assert(isinstance(owner, object))
        self.__left_index = left_index
        self.__rigth_index = right_index
        self.__owner = owner

    @property
    def LeftIndex(self):
        return self.__left_index

    @property
    def RightIndex(self):
        return self.__rigth_index

    @property
    def Owner(self):
        return self.__owner
