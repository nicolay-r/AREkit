# -*- coding: utf-8 -*-
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.helpers.news import RuSentRelNewsHelper
from core.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from core.source.rusentrel.io_utils import RuSentRelIOUtils
from core.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelNews(object):

    def __init__(self, sentences, entities):
        assert(isinstance(sentences, list))
        assert(isinstance(entities, RuSentRelDocumentEntityCollection))
        self.__sentences = sentences
        self.__entities = entities
        self.__helper = RuSentRelNewsHelper(self)

    @property
    def DocEntities(self):
        return self.__entities

    @property
    def Helper(self):
        return self.__helper

    @classmethod
    def read_document(cls, doc_id, entities):
        return RuSentRelIOUtils.read_news_file(
            doc_id=doc_id,
            process_func=lambda input_file: cls.__from_file(input_file, entities))

    @classmethod
    def __from_file(cls, input_file, entities):
        assert(isinstance(entities, RuSentRelDocumentEntityCollection))

        sentences = RuSentRelNews.__read_sentences(input_file)

        s_ind = 0
        e_ind = 0

        while s_ind < len(sentences) and e_ind < len(entities):
            e = entities.get_entity_by_index(e_ind)
            assert(isinstance(e, RuSentRelEntity))

            s = sentences[s_ind]

            if s.is_entity_goes_after(e):
                s_ind += 1
                continue

            if e in s:
                s.add_local_entity(entity=e)
                e_ind += 1
                continue

            if e.Value in [u'author', u'unknown']:
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}".format(
                e_ind,
                e.Value.encode('utf-8'), e.CharIndexBegin, e.CharIndexEnd,
                s_ind))

        assert(e_ind == len(entities))

        return cls(sentences, entities)

    @staticmethod
    def __read_sentences(input_file):
        sentences = []
        line_start = 0
        unknown_entity = u"Unknown}"

        for line in input_file.readlines():

            line = line.decode('utf-8')

            if unknown_entity in line:
                offset = line.index(unknown_entity) + len(unknown_entity)
                line_start += offset
                line = line[offset:]

            line_end = line_start + len(line) - 1

            if line != unicode('\r\n'):
                s = RuSentRelSentence(text=line,
                                      char_ind_begin=line_start,
                                      char_ind_end=line_end)
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
