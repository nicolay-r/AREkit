import pandas as pd

from feature import Feature
from core.relations import Relation
from core.annot import Entity
from core.stemmer import Stemmer
from core.news import News

# TODO: move to the environment
from core.processing.prefix import SentimentPrefixProcessor


class LexiconFeature(Feature):

    def __init__(self, csv_filepath, prefix_processor):
        assert(isinstance(prefix_processor, SentimentPrefixProcessor))
        self.stemmer = Stemmer()
        self.lexicon = pd.read_csv(csv_filepath, sep=',')
        self.prefix_processor = prefix_processor

    def __find_sentence_with_entity(self, entity, news):
        assert(isinstance(entity, Entity))
        assert(isinstance(news, News))
        entity.show()
        for i, sentence in enumerate(news.sentences):
            if sentence.has_entity(entity.ID):
                return i

        raise Exception("Can't find entity!")

    def create(self, relation):
        """ Get the sentiment sum of words between relation entities
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.find_by_ID(relation.entity_right_ID)

        s1 = self.__find_sentence_with_entity(e1, relation.news)
        s2 = self.__find_sentence_with_entity(e2, relation.news)

        scores = self.__scores(min(s1, s2), max(s1, s2), e1, e2, relation.news)

        if len(scores) == 0:
            scores.append(0)

        return [sum(scores), min(scores), max(scores)]

    def __scores(self, s_from, s_to, e1, e2, news, sentence_range=4):
        scores = [0]
        sentences = news.sentences

        # omitting relations between entities that placed so far
        if (s_to - s_from > sentence_range):
            return scores

        i = s_from + 1
        while i < s_to:
            scores += self.__get_scores(sentences[i].text)
            i += 1

        s_begin = sentences[s_from].begin

        if (s_from < s_to):
            # sentences are different
            char_from_end = sentences[s_from].end - s_begin  # [   |->......]
            char_to_begin = sentences[s_to].begin - s_begin  # [......<-|   ]
            scores += self.__get_scores(sentences[s_from].text[char_from_end:])
            scores += self.__get_scores(sentences[s_to].text[:char_to_begin])
            pass
        else:
            # single sentence
            char_to = max(e1.begin - s_begin, e2.begin - s_begin)
            char_from = min(e1.end - s_begin, e2.end - s_begin)
            scores += self.__get_scores(sentences[i].text[char_from:char_to])

        return scores

    def __get_scores(self, text):
        lemmas = self.stemmer.lemmatize_to_list(text)
        processed_lemmas = self.prefix_processor.process(lemmas)
        scores = self.__get_scores_of_processed(processed_lemmas)
        return scores

    def __get_scores_of_processed(self, processed_lemmas):
        scores = []
        signs = ['+', '-']
        sign = None
        for lemma in processed_lemmas:

            if lemma in signs:
                sign = lemma
                continue

            score = self.__get_weight(lemma)

            if (sign == '-'):
                score *= -1
                sign = None

            scores.append(score)

        return scores

    # TODO: move everything into Lexicon class
    def __get_weight(self, lemma):
        assert(type(lemma) == unicode)

        s = self.lexicon[lemma.encode('utf-8') == self.lexicon['term']]
        return s['tone'].values[0] if len(s) > 0 else 0
