from core.annot import Entity
from core.relations import Relation
from feature import Feature
from core.news import News


class PatternFeature(Feature):

    def __init__(self, patterns, max_sentence_range=5):
        assert(type(patterns) == list)
        self.patterns = patterns
        self.max_sentence_range = max_sentence_range

    # Duplicate
    def __find_sentence_with_entity(self, entity, news):
        assert(isinstance(entity, Entity))
        assert(isinstance(news, News))
        for i, sentence in enumerate(news.sentences):
            if sentence.has_entity(entity.ID):
                return i

        raise Exception("Can't find entity!")

    def create(self, relation):
        """ Get an amount of patterns between entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.find_by_ID(relation.entity_right_ID)

        s1 = self.__find_sentence_with_entity(e1, relation.news)
        s2 = self.__find_sentence_with_entity(e2, relation.news)

        return [self.__get_count(p, s1, s2, e1, e2, relation.news) for p in self.patterns]

    def __get_count(self, pattern, s_from, s_to, e1, e2, news):
        c = 0
        sentences = news.sentences

        # omitting relations between entities that placed so far
        if (s_to - s_from > self.max_sentence_range):
            return c

        i = s_from + 1
        while i < s_to:
            c += sentences[i].text.count(pattern)
            i += 1

        s_begin = sentences[s_from].begin

        if (s_from < s_to):
            # sentences are different
            char_from_end = sentences[s_from].end - s_begin  # [   |->......]
            char_to_begin = sentences[s_to].begin - s_begin  # [......<-|   ]
            c += sentences[s_from].text[char_from_end:].count(pattern)
            c += sentences[s_to].text[:char_to_begin].count(pattern)
            pass
        else:
            # single sentence
            char_to = max(e1.begin - s_begin, e2.begin - s_begin)
            char_from = min(e1.end - s_begin, e2.end - s_begin)
            c += sentences[s_from].text[char_from:char_to].count(pattern)

        return c
