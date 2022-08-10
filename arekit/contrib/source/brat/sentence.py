from arekit.common.bound import Bound
from arekit.common.news.sentence import BaseNewsSentence


class BratSentence(BaseNewsSentence):
    """ Represent a raw sentence of BRAT.
        Provides text could be used to parse then.
        Provides API to store entities.
    """

    def __init__(self, text, char_ind_begin, char_ind_end, entities):
        """ entities: list of BratEntities
        """
        assert(isinstance(text, str) and len(text) > 0)
        assert(isinstance(char_ind_begin, int))
        assert(isinstance(char_ind_end, int))
        assert(isinstance(entities, list))

        super(BratSentence, self).__init__(text=text)
        self.__begin = char_ind_begin
        self.__end = char_ind_end
        self.__entities = entities

    def iter_entity_with_local_bounds(self, avoid_intersection=True):
        last_position = -1

        for entity in self.__entities:
            start = entity.CharIndexBegin - self.__begin
            end = entity.CharIndexEnd - self.__begin

            if start <= last_position and avoid_intersection:
                # intersected with the previous one.
                continue

            yield entity, Bound(pos=start, length=end - start)
            last_position = end
