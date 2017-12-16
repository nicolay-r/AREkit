import numpy as np
import core.env as env
from core.source.lexicon import Lexicon
from core.relations import Relation
from base import Base


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
        p_before = env.stemmer.analyze_pos_list(l_before)
        return [1 if p in p_before else 0 for p in self.POS]
