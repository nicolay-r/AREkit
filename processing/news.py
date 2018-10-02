from core.source.entity import Entity
from core.processing.stemmer import Stemmer


class NewsProcessor:

    def __init__(self, news, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.news = news
        self.stemmer = stemmer
        self.lemmatized_sentences = self._process(news)
        self.sentence_by_entity = self._index_sentence_by_entity(news)
        self.words_count = self._get_words_count()

    @property
    def sentences(self):
        return self.news.sentences

    def _get_words_count(self):
        result = 0
        for s in self.lemmatized_sentences.itervalues():
            result += len(s)
        return result

    def get_text_between_entities_to_lemmatized_list(self, e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        e1, e2 = self._check_and_order_entities(e1, e2)

        s1 = self.get_sentence_by_entity(e1)
        s2 = self.get_sentence_by_entity(e2)

        if s1.index == s2.index:
            return self.stemmer.lemmatize_to_list(s1.text[e1.end-s1.begin:e2.begin-s1.begin])

        text = []
        text += self.stemmer.lemmatize_to_list(s1.text[e1.end-s1.begin:])
        for i in range(s1.index+1, s2.index):
            text += self.lemmatized_sentences[i]
        text += self.stemmer.lemmatize_to_list(s2.text[:e2.begin-s2.begin])

        return text

    def get_text_between_sentence_bounds(self, sentence, left_bound, right_bound):
        assert(isinstance(left_bound, int))
        assert(isinstance(right_bound, int))
        assert(left_bound <= right_bound)
        return self.stemmer.lemmatize_to_list(
            sentence.text[left_bound-sentence.begin:right_bound-sentence.begin])

    def get_text_before_entity_as_str(self, e):
        assert(isinstance(e, Entity))
        s = self.get_sentence_by_entity(e)

        # TODO. Copied from below
        texts = []
        for i in range(0, s.index):
            texts.append(self.sentences[i].text)

        texts.append(s.text[:e.begin-s.begin])

        return u" ".join(texts)

    def get_text_between_entities_to_str(self, e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        e1, e2 = self._check_and_order_entities(e1, e2)

        s1 = self.get_sentence_by_entity(e1)
        s2 = self.get_sentence_by_entity(e2)

        if (s1.index == s2.index):
            return s1.text[e1.end-s1.begin: e2.begin-s1.begin]

        texts = []
        texts.append(s1.text[e1.end-s1.begin:])
        for i in range(s1.index+1, s2.index):
            texts.append(self.sentences[i].text)
        texts.append(s2.text[:e2.begin-s2.begin])

        return u" ".join(texts)

    def get_lemmas_after_entity_to_list(self, e):
        assert(isinstance(e, Entity))
        s = self.get_sentence_by_entity(e)
        text = s.text[e.end-s.begin:]
        return self.stemmer.lemmatize_to_list(text)

    def get_lemmas_before_entity_to_list(self, e):
        assert(isinstance(e, Entity))
        s = self.get_sentence_by_entity(e)
        text = s.text[:e.begin-s.begin]
        return self.stemmer.lemmatize_to_list(text)

    def get_sentence_by_entity(self, e):
        assert(isinstance(e, Entity))
        if (e.ID not in self.sentence_by_entity):
            print "FAILED: {} key wasn't found".format(e.value.encode('utf-8'))
        return self.news.sentences[self.sentence_by_entity[e.ID]]

    @staticmethod
    def _check_and_order_entities(e1, e2):
        if e1.get_int_ID() == e2.get_int_ID():
            raise Exception("Entities are equal!, {}->{}".format(
                e1.value.encode('utf-8'), e2.value.encode('utf-8')))
            return None

        if e1.get_int_ID() > e2.get_int_ID():
            return e2, e1
        return e1, e2

    def _process(self, news):
        lemmatized_sentences = {}
        for s in news.sentences:
            lemmatized_sentences[s.index] = self.stemmer.lemmatize_to_list(s.text)
        return lemmatized_sentences

    @staticmethod
    def _index_sentence_by_entity(news):
        index = {}
        for s in news.sentences:
            for e_ID in s.entities_ids:
                index[e_ID] = s.index
        return index
