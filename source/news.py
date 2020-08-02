# -*- coding: utf-8 -*-

import io
from core.helpers.news import NewsHelper
from core.source.entity import EntityCollection


# TODO. Class should be in rusentrel/news.py
class News:

    def __init__(self, sentences, entities):
        assert(isinstance(sentences, list))
        assert(isinstance(entities, EntityCollection))
        self.__sentences = sentences
        self.__entities = entities
        self.__helper = NewsHelper(self)

    @property
    def Entities(self):
        return self.__entities

    @property
    def Helper(self):
        return self.__helper

    @classmethod
    def from_file(cls, filepath, entities):
        """ Read news from file
        """
        assert(isinstance(filepath, str))
        assert(isinstance(entities, EntityCollection))

        sentences = News.read_sentences(filepath)

        s_ind = 0
        e_ind = 0

        while s_ind < len(sentences) and e_ind < len(entities):
            e = entities.get_entity_by_index(e_ind)
            s = sentences[s_ind]

            if e.begin > s.End:
                s_ind += 1
                continue

            if e.begin >= s.Begin and e.end <= s.End:
                s.add_local_entity(id=e.ID,
                                   begin=e.begin - s.Begin,
                                   end=e.end - s.Begin)
                e_ind += 1
                continue

            if e.value in ['author', 'unknown']:
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}  s({},{})".format(
                e_ind,
                e.value.encode('utf-8'), e.begin, e.end,
                s_ind,
                s.Begin, s.End))

        assert(e_ind == len(entities))

        return cls(sentences, entities)

    @staticmethod
    def read_sentences(filepath):
        assert(isinstance(filepath, str))

        with io.open(filepath, 'rt', newline='\n', encoding='utf-8') as f:

            sentences = []
            line_start = 0
            unknown_entity = "Unknown}"

            for line in f.readlines():

                if unknown_entity in line:
                    offset = line.index(unknown_entity) + len(unknown_entity)
                    line_start += offset
                    line = line[offset:]

                line_end = line_start + len(line) - 1

                if line != str('\r\n'):
                    s = Sentence(text=line,
                                 begin=line_start,
                                 end=line_end)
                    sentences.append(s)

                line_start = line_end + 1

        return sentences

    def sentences_count(self):
        return len(self.__sentences)

    def get_sentence_by_index(self, index):
        return self.__sentences[index]

    def iter_sentences(self):
        for sentence in self.__sentences:
            yield sentence


# TODO. Class should be in rusentrel/sentence.py
class Sentence:

    def __init__(self, text, begin, end):
        assert(isinstance(text, str) and len(text) > 0)
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.__text = text
        self.__entity_info = []
        self.__begin = begin
        self.__end = end

    @property
    def Begin(self):
        return self.__begin

    @property
    def End(self):
        return self.__end

    @property
    def Text(self):
        return self.__text

    def add_local_entity(self, id, begin, end):
        assert(isinstance(id, str))
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.__entity_info.append((id, begin, end))

    def iter_entity_ids(self):
        for entity in self.__entity_info:
            yield entity[0]  # ID

    def iter_entities_info(self):
        for info in self.__entity_info:
            yield info
