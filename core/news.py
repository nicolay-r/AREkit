# -*- coding: utf-8 -*-
import io


class News:

    def __init__(self, sentences, entities):
        self.sentences = sentences
        self.entities = entities

    @staticmethod
    def from_file(filepath, entities):
        """ Read news from file
        """

        with io.open(filepath, 'rt', newline='\n', encoding='utf-8') as f:
            sentences = []
            paragraph_id = 0
            line_start = 0
            line_end = 0

            for line in f.readlines():
                line_end = line_start + len(line) - 1

                if (line == unicode('\r\n')):
                    paragraph_id += 1
                else:
                    s = Sentence(line, paragraph_id, line_start, line_end)
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

            raise Exception("e_i:{} e:({},{}), s_i:{}  s({},{})".format(
                e_ind, e.begin, e.end, s_ind, s.begin, s.end))

        return News(sentences, entities)

    def show(self):
        for s in self.sentences:
            print s.text.encode('utf-8')
            for e in s.entities:
                ID, begin, end = e
                print "\t - {}, ({},{}) = {}".format(
                    ID, begin, end,
                    s.text[begin:end].encode('utf-8'))


class Sentence:

    def __init__(self, text, paragraph_id, begin, end):
        assert(type(text) == unicode and len(text) > 0)
        assert(type(paragraph_id) == int)
        assert(type(begin) == int)
        assert(type(end) == int)

        self.text = text
        self.paragraph_id = paragraph_id
        self.entity_ids = []
        self.begin = begin
        self.end = end

    def add_entity(self, ID, begin, end):
        """ Local entity indices
        """
        assert(type(ID) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        self.entity_ids.append((ID, begin, end))

    def has_entity(self, entity_ID):
        for e in self.entity_ids:
            if e[0] == entity_ID:
                return True
        return False

    @property
    def entities(self):
        for e in self.entity_ids:
            yield e
