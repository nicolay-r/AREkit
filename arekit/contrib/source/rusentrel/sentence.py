from arekit.common.bound import Bound
from arekit.common.news.sentence import BaseNewsSentence
from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity


# TODO. #287 to brat.
class RuSentRelSentence(BaseNewsSentence):
    """ Represent a raw sentence of rusentrel.
        Provides text could be used to parse then.
        Provides API to store entites.
    """

    # TODO. Remove begin / end parameters
    def __init__(self, text, char_ind_begin, char_ind_end):
        assert(isinstance(text, str) and len(text) > 0)
        assert(isinstance(char_ind_begin, int))
        assert(isinstance(char_ind_end, int))

        super(RuSentRelSentence, self).__init__(text=text)
        self.__begin = char_ind_begin
        self.__end = char_ind_end
        self.__entities = []

    # region public methods

    def add_local_entity(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        self.__entities.append(entity)

    def iter_entity_with_local_bounds(self):
        for entity in self.__entities:
            start = entity.CharIndexBegin - self.__begin
            end = entity.CharIndexEnd - self.__begin
            yield entity, Bound(pos=start, length=end - start)

    def is_entity_goes_after(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return entity.CharIndexBegin > self.__end

    # endregion

    # region overriden methods

    def __contains__(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return entity.CharIndexBegin >= self.__begin and entity.CharIndexEnd <= self.__end

    # endregion
