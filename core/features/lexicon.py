import pandas as pd

from feature import Feature
from core.relations import Relation
from core.annot import Entity
from core.stemmer import Stemmer
from core.processing.prefix import SentimentPrefixProcessor


class LexiconFeature(Feature):

    def __init__(self, csv_filepath):
        self.stemmer = Stemmer()
        self.df = pd.read_csv(csv_filepath)

    def __find_sentence_with_entity(self, entity):
        assert(isinstance(entity, Entity))
        for i, sentence in enumerate(self.news):
            if sentence.has_entity(entity):
                return i

        raise Exception("Can't find entity!")

    def create(self, relation):
        """ Get the sentiment sum of words between relation entities
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.find_by_ID(relation.entity_right_ID)

        s1 = self.__find_sentence_with_entity(e1)
        s2 = self.__find_sentence_with_entity(e2)

        scores = self.__get_scores(min(s1, s2), max(s1, s2))

        return [sum(scores), min(scores), max(scores)]

    def __get_scores(self, s_from, s_to, e1, e2, f):
        i = s_from

        # TODO: full sentences
        while i+1 < s_to:
            pass

        if (s_from < s_to):
            pass
        else:
            # single sentence
            pass

    def __get_lemmas(self, text):
        # lemmas = self.stemmer.lemmatize_to_list(text)
        # TODO: apply sentiment prefix processor
        pass
