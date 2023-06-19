from functools import cmp_to_key

from arekit.common.bound import Bound
from arekit.common.docs.sentence import BaseDocumentSentence
from arekit.contrib.source.brat.entities.compound import BratCompoundEntity
from arekit.contrib.source.brat.entities.entity import BratEntity


class BratSentence(BaseDocumentSentence):
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

    @staticmethod
    def cmp_entities(a, b):
        assert(isinstance(a, BratEntity))
        assert(isinstance(b, BratEntity))
        if a.IndexBegin != b.IndexBegin:
            # Ordered by appearance
            return a.IndexBegin - b.IndexBegin
        else:
            # Ordered by length first
            b_length = b.IndexEnd - b.IndexBegin
            a_length = a.IndexEnd - a.IndexBegin
            return b_length - a_length

    def iter_entity_with_local_bounds(self):
        self.__entities.sort(key=cmp_to_key(lambda a, b: self.cmp_entities(a, b)))

        bounds_and_entities = []

        # Merging nested entities.
        for entity in self.__entities:
            start = entity.IndexBegin - self.__index_begin
            end = entity.IndexEnd - self.__index_begin
            bound = Bound(pos=start, length=end - start)

            updated = False
            if len(bounds_and_entities) > 0:
                last_bound, last_entities = bounds_and_entities[-1]
                if bound.itersects_with(last_bound):
                    # Update.
                    last_entities.append(entity)
                    bounds_and_entities[-1] = (bound.intersect(last_bound), last_entities)
                    updated = True

            if not updated:
                bounds_and_entities.append((bound, [entity]))

        # Returning result.
        for item in bounds_and_entities:
            bound, entities = item
            entity = entities[0] if len(entities) == 1 else \
                BratCompoundEntity.from_list(root=entities[0], childs=entities[1:])
            yield entity, bound
