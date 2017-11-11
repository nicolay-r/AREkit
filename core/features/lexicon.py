import numpy as np

from core.processing.prefix import SentimentPrefixProcessor
from core.source.lexicon import Lexicon
from core.relations import Relation
from base import Base


class LexiconFeature(Base):

    def __init__(self, lexicon, prefix_processor, max_sentence_range=4):
        assert(isinstance(lexicon, Lexicon))
        assert(isinstance(prefix_processor, SentimentPrefixProcessor))
        self.lexicon = lexicon
        self.prefix_processor = prefix_processor
        self.max_sentence_range = max_sentence_range

    def create(self, relation):
        """ Get the sentiment sum of words between relation entities
        """
        assert(isinstance(relation, Relation))

        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)

        s_ind1 = relation.news.get_sentence_by_entity(e1).index
        s_ind2 = relation.news.get_sentence_by_entity(e2).index

        scores = []
        if abs(s_ind1 - s_ind2) < 3:
            lemmas = relation.news.Processed.get_text_between_entities_to_lemmatized_list(e1, e2)
            scores = self._get_scores_of_processed(lemmas)

        if len(scores) == 0:
            scores.append(0)

        p_s_all = float(sum(scores))/len(scores)
        p_s_max = float(sum(s > 0 for s in scores))/len(scores)
        p_s_min = float(sum(s < 0 for s in scores))/len(scores)

        return np.array([p_s_all, p_s_max, p_s_min])

    def _get_scores_of_processed(self, processed_lemmas):
        scores = []
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
    def _show_lemmas(processed_lemmas):
        for l in processed_lemmas:
            print l.encode('utf-8'),
        if len(processed_lemmas) > 0:
            print '\n'
