import numpy as np

import core.env as env
from base import Base
from core.runtime.relations import Relation
from core.source.lexicon import Lexicon


class ContextSentimentAfterFeature(Base):

    LIMIT = 2

    def __init__(self, lexicon):
        assert(isinstance(lexicon, Lexicon))
        self.lexicon = lexicon

    def create(self, relation):
        """ Sentiment context features
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)

        s_ind1 = relation.news.get_sentence_by_entity(e1).index
        s_ind2 = relation.news.get_sentence_by_entity(e2).index

        if s_ind1 != s_ind2:
            return [float(0)] * 3

        l_after = relation.news.Processed.get_lemmas_after_entity_to_list(e1)[:self.LIMIT]
        scores = [self.lexicon.get_score(l) for l in l_after]

        if len(l_after) == 0:
            scores.append(0)

        # print scores

        average = float(sum(scores))/len(scores)
        return np.array([average, max(scores), min(scores)])

    def feature_names(self):
        name = self.__class__.__name__
        return [name + '_avg', name + '_max', name + '_min']


class ContextPosBeforeFeature(Base):
    """ https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
    """
    LIMIT = 2
    POS = ['pr', 's', 'adv', 'conj', 'v', 'num']

    def __init__(self):
        pass

    def create(self, relation):
        """ Part of speech features
        """
        assert(isinstance(relation, Relation))

        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)

        s_ind1 = relation.news.get_sentence_by_entity(e1).index
        s_ind2 = relation.news.get_sentence_by_entity(e2).index

        if s_ind1 != s_ind2:
            return [0] * len(self.POS)

        pos_vector = self._create_vector(relation.news, e1)
        # print pos_vector
        return np.array(pos_vector)

    def feature_names(self):
        name = self.__class__.__name__
        return [name + '_' + p for p in self.POS]

    def _create_vector(self, news, e):
        l_before = news.Processed.get_lemmas_before_entity_to_list(e)[-self.LIMIT:]
        p_before = env.stemmer.get_terms_pos(l_before)
        return [1 if p in p_before else 0 for p in self.POS]


class ContextFeature(Base):
    """ Feature that provides a sentiment information of a context before/after/inner of a relation.
    """

    def __init__(self, lexicon):
        assert(isinstance(lexicon, Lexicon))
        self.lexicon = lexicon

    def calculate(self, relations):
        """ functions_list: np.min, np.max, np.average
        """
        assert(type(relations) == list)
        results = [[0, 0, 0]]

        for relation in relations:
            e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
            e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
            if relation.news.get_sentence_by_entity(e1) == relation.news.get_sentence_by_entity(e2):
                results.append(self.create(relation))

        return self._normalize(
            np.concatenate((
                np.min(results, axis=0),
                np.max(results, axis=0),
                np.average(results, axis=0))))

    def create(self, relation):
        """ Sentiment context features
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)

        s = relation.news.get_sentence_by_entity(e1)

        e1, e2 = self._order_entities(e1, e2)

        e_before = relation.news.entities.get_previous_entity(e1)
        e_after = relation.news.entities.get_next_entity(e2)

        has_entity_before_in_sentence = e_before is not None and relation.news.get_sentence_by_entity(e_before).index == s.index
        has_entity_after_in_sentence = e_after is not None and relation.news.get_sentence_by_entity(e_after).index == s.index

        b_left = s.begin if not has_entity_before_in_sentence else e_before.end
        b_right = s.end if not has_entity_after_in_sentence else e_after.begin

        l_inside = relation.news.Processed.get_text_between_entities_to_lemmatized_list(e1, e2)
        l_before = relation.news.Processed.get_text_between_sentence_bounds(s, b_left, e1.begin)
        l_after = relation.news.Processed.get_text_between_sentence_bounds(s, e2.end, b_right)

        scores = [sum(self._get_scores_of_processed(l_before)),
                  sum(self._get_scores_of_processed(l_inside)),
                  sum(self._get_scores_of_processed(l_after))]

        # self._debug_show(l_before)
        # print " _{}_ ".format(e1.value.encode('utf-8')),
        # self._debug_show(l_inside)
        # print " _{}_ ".format(e2.value.encode('utf-8')),
        # self._debug_show(l_after)
        # print scores

        return np.array(scores)

    def feature_names(self):
        name = self.__class__.__name__
        return [name + '_before', name + '_inside', name + '_after']

    def _get_scores_of_processed(self, processed_lemmas):
        # TODO. Tha same code in lexicon feature
        scores = [0]
        signs = ['+', '-']
        sign = None
        for lemma in processed_lemmas:

            if lemma in signs:
                sign = lemma
                continue

            score = self.lexicon.get_score(lemma)

            if sign == '-':
                score *= -1
                sign = None

            scores.append(score)

        return scores

    @staticmethod
    def _order_entities(e1, e2):
        if e1.begin > e2.begin:
            return e2, e1
        return e1, e2

    def _debug_show(self, lemmas):
        for l in lemmas:
            print l, ' ',

