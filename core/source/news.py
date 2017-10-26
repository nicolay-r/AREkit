# -*- coding: utf-8 -*-

import io
from core.source.entity import Entity


class News:

    def __init__(self, sentences, entities):
        self.sentences = sentences
        self.entities = entities
        self.sentence_by_entity = self._index_sentence_by_entity()

    @staticmethod
    def from_file(filepath, entities):
        """ Read news from file
        """

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

        while (s_ind < len(sentences) and e_ind < entities.count()):
            e = entities.get(e_ind)
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

        return News(sentences, entities)

    def get_sentence_by_entity(self, entity):
        assert(isinstance(entity, Entity))
        return self.sentences[self.sentence_by_entity[entity.ID]]

    def _index_sentence_by_entity(self):
        index = {}
        for s in self.sentences:
            for e_ID in s.entities_ids:
                index[e_ID] = s.index
        return index

    def show(self):
        for s in self.sentences:
            print s.text.encode('utf-8')
            for e in s.entities:
                ID, begin, end = e
                print "\t - {}, ({},{}) = {}".format(
                    ID, begin, end, s.text[begin:end].encode('utf-8'))


class Sentence:

    def __init__(self, text, paragraph_id, begin, end, index):
        assert(type(text) == unicode and len(text) > 0)
        assert(type(paragraph_id) == int)
        assert(type(begin) == int)
        assert(type(end) == int)
        assert(type(index) == int)

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
        assert(type(ID) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        self.entity_info.append((ID, begin, end))
        self.entity_set_ids.add(ID)

    def has_entity(self, entity_ID):
        assert(type(entity_ID == unicode))
        return entity_ID in self.entity_set_ids

    @property
    def entities(self):
        for e in self.entity_info:
            yield e

    @property
    def entities_ids(self):
        for e in self.entity_info:
            yield e[0]  # ID
