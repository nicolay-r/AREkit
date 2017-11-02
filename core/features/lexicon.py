import pandas as pd

from feature import Feature
from core.source.relations import Relation
from core.stemmer import Stemmer

# TODO: move to the environment
from core.processing.prefix import SentimentPrefixProcessor


class LexiconFeature(Feature):

    def __init__(self, csv_filepath, prefix_processor, max_sentence_range=4):
        assert(isinstance(prefix_processor, SentimentPrefixProcessor))
        self.stemmer = Stemmer()
        self.lexicon = pd.read_csv(csv_filepath, sep=',')
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

        return self._normalize([p_s_all, p_s_max, p_s_min])

    def _get_scores_of_processed(self, processed_lemmas):
        scores = []
        signs = ['+', '-']
        sign = None
        for lemma in processed_lemmas:

            if lemma in signs:
                sign = lemma
                continue

            score = self._get_weight(lemma)

            if (sign == '-'):
                score *= -1
                sign = None

            scores.append(score)

        return scores

    # TODO: move everything into Lexicon class
    def _get_weight(self, lemma):
        assert(type(lemma) == unicode)

        s = self.lexicon[lemma.encode('utf-8') == self.lexicon['term']]
        return s['tone'].values[0] if len(s) > 0 else 0

    @staticmethod
    def _show_lemmas(processed_lemmas):
        for l in processed_lemmas:
            print l.encode('utf-8'),
        if len(processed_lemmas) > 0:
            print '\n'
