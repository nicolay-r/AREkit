from arekit.common.bound import Bound
from arekit.common.news.sentence import BaseNewsSentence


class BratSentence(BaseNewsSentence):
    """ Represent a raw sentence of BRAT.
        Provides text could be used to parse then.
        Provides API to store entities.
    """

    def __init__(self, text, index_begin, entities):
        """ entities: list of BratEntities
            index_begin: int
                - char index (in case of string type of `text`)
                - term index (in case of list type of `text`)
        """
        assert(isinstance(text, str) or isinstance(text, list))
        assert(isinstance(index_begin, int))
        assert(isinstance(entities, list))
        super(BratSentence, self).__init__(text=text)
        self.__index_begin = index_begin
        self.__entities = entities

    def iter_entity_with_local_bounds(self, avoid_intersection=True):
        last_position = -1

        for entity in self.__entities:
            start = entity.IndexBegin - self.__index_begin
            end = entity.IndexEnd - self.__index_begin

            if start <= last_position and avoid_intersection:
                # intersected with the previous one.
                continue

            yield entity, Bound(pos=start, length=end - start)
            last_position = end
