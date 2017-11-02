from core.source.entity import Entity
import core.environment as env


class NewsProcessor:

    def __init__(self, news):
        self.news = news
        self.lemmatized_sentences = self._process(news)
        self.sentence_by_entity = self._index_sentence_by_entity(news)

    @property
    def sentences(self):
        return self.news.sentences

    def get_text_between_entities_to_lemmatized_list(self, e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        e1, e2 = self._check_and_order_entities(e1, e2)

        s1 = self.get_sentence_by_entity(e1)
        s2 = self.get_sentence_by_entity(e2)

        if (s1.index == s2.index):
            return self._lemmatize_to_list(s1.text[e1.end-s1.begin:e2.begin-s1.begin])

        text = []
        text += self._lemmatize_to_list(s1.text[e1.end-s1.begin:])
        for i in range(s1.index+1, s2.index):
            text += self.lemmatized_sentences[i]
        text += self._lemmatize_to_list(s2.text[:e2.begin-s2.begin])

        return text

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

    def get_lemmas_after_entity(self, e):
        assert(isinstance(e, Entity))
        pass

    def get_lemmas_before_entity(self, e):
        assert(isinstance(e, Entity))
        pass

    def get_sentence_by_entity(self, e):
        assert(isinstance(e, Entity))
        if (e.ID not in self.sentence_by_entity):
            print "FAILED: {} key wasn't found".format(e.value.encode('utf-8'))
        return self.news.sentences[self.sentence_by_entity[e.ID]]

    @staticmethod
    def _check_and_order_entities(e1, e2):
        if (e1.get_int_ID() == e2.get_int_ID()):
            raise Exception("Entities are equal!, {}->{}".format(
                e1.value.encode('utf-8'), e2.value.encode('utf-8')))
            return None

        if (e1.get_int_ID() > e2.get_int_ID()):
            return e2, e1
        return e1, e2

    @staticmethod
    def _process(news):
        lemmatized_sentences = {}
        for s in news.sentences:
            lemmatized_sentences[s.index] = NewsProcessor._lemmatize_to_list(s.text)
        return lemmatized_sentences

    @staticmethod
    def _lemmatize_to_list(text):
        return env.stemmer.lemmatize_to_list(text)

    @staticmethod
    def _index_sentence_by_entity(news):
        index = {}
        for s in news.sentences:
            for e_ID in s.entities_ids:
                index[e_ID] = s.index
        return index
