# -*- coding: utf-8 -*-

import io
from core.processing.lemmatization.base import Stemmer
from core.processing.news import NewsProcessor
from core.source.entity import EntityCollection


class News:

    def __init__(self, sentences, entities, stemmer):
        self.entities = entities
        self.sentences = sentences
        self.processed = NewsProcessor(self, stemmer)

    def get_sentences(self):
        for s in self.sentences:
            yield s

    def get_words_count(self):
        count = self.processed.words_count
        for e in self.entities:
            words_in_entities = len(e.value.split(' '))
            count -= (words_in_entities - 1)
        return count

    def get_entities(self):
        return self.entities

    @property
    def Processed(self):
        return self.processed

    @classmethod
    def from_file(cls, filepath, entities, stemmer):
        """ Read news from file
        """
        assert(isinstance(filepath, unicode) or isinstance(filepath, unicode))
        assert(isinstance(entities, EntityCollection))
        assert(isinstance(stemmer, Stemmer))

        with io.open(filepath, 'rt', newline='\n', encoding='utf-8') as f:

            sentences = []
            paragraph_id = 0
            line_start = 0
            line_end = 0
            s_ind = 0

            for line in f.readlines():
                line_end = line_start + len(line) - 1

                if (line == unicode('\r\n')):
                    paragraph_id += 1
                else:
                    s = Sentence(line, paragraph_id, line_start, line_end,
                                 s_ind)
                    s_ind += 1
                    sentences.append(s)

                line_start = line_end + 1

        s_ind = 0
        e_ind = 0

        while s_ind < len(sentences) and e_ind < entities.count():
            e = entities.get_entity_by_index(e_ind)
            s = sentences[s_ind]

            if e.begin > s.end:
                s_ind += 1
                continue

            if e.begin >= s.begin and e.end <= s.end:
                s.add_entity(e.ID, e.begin - s.begin, e.end - s.begin)
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}  s({},{})".format(
                e_ind,
                e.value.encode('utf-8'), e.begin, e.end,
                s_ind,
                s.begin, s.end))

        assert(e_ind == entities.count())

        return cls(sentences, entities, stemmer)

    def get_sentence_by_entity(self, entity):
        return self.processed.get_sentence_by_entity(entity)

    def show(self):
        for s in self.sentences:
            print s.text.encode('utf-8')
            for e in s.entities:
                ID, begin, end = e
                print "\t - {}, ({},{}) = {}".format(
                    ID, begin, end, s.text[begin:end].encode('utf-8'))


class Sentence:

    def __init__(self, text, paragraph_id, begin, end, index):
        assert(isinstance(text, unicode) and len(text) > 0)
        assert(isinstance(paragraph_id, int))
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        assert(isinstance(index, int))

        self.text = text
        self.paragraph_id = paragraph_id
        self.entity_info = []
        self.entity_set_ids = set()
        self.begin = begin
        self.end = end
        self.index = index

    def add_entity(self, ID, begin, end):
        """ Local entity indices
        """
        assert(isinstance(ID, unicode))
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.entity_info.append((ID, begin, end))
        self.entity_set_ids.add(ID)

    def has_entity(self, entity_ID):
        assert(isinstance(entity_ID, unicode))
        return entity_ID in self.entity_set_ids

    @property
    def entities(self):
        for e in self.entity_info:
            yield e

    @property
    def entities_ids(self):
        for e in self.entity_info:
            yield e[0]  # ID
