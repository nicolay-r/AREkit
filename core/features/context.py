import numpy as np
import pandas as pd

import core.environment as env
from core.relations import Relation
from feature import Feature


class ContextSentimentAfterFeature(Feature):

    LIMIT = 2

    # TODO: move everything into Lexicon class
    def __init__(self, csv_filepath):
        self.lexicon = pd.read_csv(csv_filepath, sep=',')

    def create(self, relation):
        """ Sentiment context features
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)

        s_ind1 = relation.news.get_sentence_by_entity(e1).index
        s_ind2 = relation.news.get_sentence_by_entity(e2).index

        if s_ind1 != s_ind2:
            return [float(0)] * 3

        l_after = relation.news.Processed.get_lemmas_after_entity_to_list(e1)[:self.LIMIT]
        scores = self._get_scores(l_after)

        if len(l_after) == 0:
            scores.append(0)

        # print scores

        average = float(sum(scores))/len(scores)
        return np.array([average, max(scores), min(scores)])

    # TODO: move into lexicon class
    def _get_scores(self, lemmas):
        scores = []
        for l in lemmas:
            score = self._get_weight(l)
            scores.append(score)
        return scores

    # TODO: move everything into Lexicon class
    def _get_weight(self, lemma):
        assert(type(lemma) == unicode)
        s = self.lexicon[lemma.encode('utf-8') == self.lexicon['term']]
        return s['tone'].values[0] if len(s) > 0 else 0


class ContextPosBeforeFeature(Feature):
    """ https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
    """
    POS = ['pr', 's', 'adv', 'conj', 'v', 'num']

    def __init__(self):
        pass

    def create(self, relation):
        """ Part of speech features
        """
        assert(isinstance(relation, Relation))

        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)

        s_ind1 = relation.news.get_sentence_by_entity(e1).index
        s_ind2 = relation.news.get_sentence_by_entity(e2).index

        if s_ind1 != s_ind2:
            return [0] * len(self.POS)

        pos_vector = self._create_vector(relation.news, e1)
        # print pos_vector
        return np.array(pos_vector)

    def _create_vector(self, news, e, limit=2):
        l_before = news.Processed.get_lemmas_before_entity_to_list(e)[-limit:]
        p_before = env.stemmer.analyze_pos_list(l_before)
        return [1 if p in p_before else 0 for p in self.POS]
