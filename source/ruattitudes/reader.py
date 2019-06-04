# -*- coding: utf-8 -*-
from core.evaluation.labels import Label
from core.processing.lemmatization.base import Stemmer
from core.runtime.object import TextObject
from core.runtime.parser import TextParser
from core.runtime.ref_opinon import RefOpinion
from core.source.ruattitudes.news import ProcessedNews
from core.source.ruattitudes.sentence import ProcessedSentence


# TODO. Rename as RuAttitudesFormatReader.
class ContextsReader(object):

    NEWS_SEP_KEY = u'--------'
    FILE_KEY = u"Файл:"
    OBJ_KEY = u"Oбъект:"
    TITLE_KEY = u"Заголовок:"
    SINDEX_KEY = u"Предложение:"
    OPINION_KEY = u"Отношение:"
    STEXT_KEY = u"Текст:"
    TEXT_IND_KEY = u"Номер:"
    TERMS_IN_TITLE = u"Термов в заголовке:"
    TERMS_IN_TEXT = u"Термов в тексте:"
    FRAMEVAR_TITLE = u"Вариант фрейма:"

    def __iter__(self):
        pass

    @staticmethod
    def iter_processed_news(filepath, stemmer=None):
        assert(isinstance(filepath, unicode))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        reset = False
        title = None
        title_terms_count = None
        text_terms_count = None
        processed_sentences = []
        opinions_list = []
        objects_list = []
        s_index = 0
        news_index = None

        with open(filepath, 'r') as input:
            for line in input.readlines():
                line = line.decode('utf-8')

                if ContextsReader.FILE_KEY in line:
                    pass

                if ContextsReader.OBJ_KEY in line:
                    object = ContextsReader.__parse_object(line)
                    objects_list.append(object)

                if ContextsReader.OPINION_KEY in line:
                    opinion = ContextsReader.__parse_opinion(line, objects_list=objects_list)
                    opinions_list.append(opinion)

                if ContextsReader.FRAMEVAR_TITLE in line:
                    pass

                if ContextsReader.TERMS_IN_TITLE in line:
                    title_terms_count = ContextsReader.__parse_terms_in_title_count(line)

                if ContextsReader.TERMS_IN_TEXT in line:
                    text_terms_count = ContextsReader.__parse_terms_in_text_count(line)

                if ContextsReader.SINDEX_KEY in line:
                    s_index = ContextsReader.__parse_sentence_index(line)

                if ContextsReader.TEXT_IND_KEY in line:
                    news_index = ContextsReader.__parse_text_index(line)

                if ContextsReader.TITLE_KEY in line:
                    title = ProcessedSentence(is_title=True,
                                              parsed_text=ContextsReader.__parse_sentence(line,
                                                                                          is_title=True,
                                                                                          stemmer=stemmer),
                                              ref_opinions=opinions_list,
                                              objects_list=objects_list,
                                              sentence_index=-1)
                    processed_sentences.append(title)
                    assert(title_terms_count == len(title.ParsedText) or title_terms_count is None)
                    reset = True

                if ContextsReader.STEXT_KEY in line:
                    sentence = ProcessedSentence(is_title=False,
                                                 parsed_text=ContextsReader.__parse_sentence(line,
                                                                                             is_title=False,
                                                                                             stemmer=stemmer),
                                                 ref_opinions=opinions_list,
                                                 objects_list=objects_list,
                                                 sentence_index=s_index)
                    processed_sentences.append(sentence)
                    assert(text_terms_count == len(sentence.ParsedText) or text_terms_count is None)
                    reset = True

                if ContextsReader.NEWS_SEP_KEY in line and title is not None:
                    yield ProcessedNews(processed_sentences=processed_sentences,
                                        news_index=news_index)
                    processed_sentences = []
                    reset = True

                if reset:
                    opinions_list = []
                    objects_list = []
                    title_terms_count = None
                    reset = False

        if len(processed_sentences) > 0:
            yield ProcessedNews(processed_sentences=processed_sentences,
                                news_index=news_index)
            processed_sentences = []

        assert(len(processed_sentences) == 0)

    @staticmethod
    def __parse_opinion(line, objects_list):
        assert(isinstance(objects_list, list))

        line = line[len(ContextsReader.OPINION_KEY):]

        s_from = line.index(u'b:(')
        s_to = line.index(u')', s_from)
        label = Label.from_int(int(line[s_from+3:s_to]))

        o_from = line.index(u'oi:[')
        o_to = line.index(u']', o_from)
        left_object_id, right_object_id = line[o_from+4:o_to].split(u',')

        left_object_id = int(left_object_id)
        right_object_id = int(right_object_id)

        ref_opinion = RefOpinion(left_index=left_object_id,
                                 right_index=right_object_id,
                                 sentiment=label,
                                 owner=objects_list)

        s_from = line.index(u'si:{')
        s_to = line.index(u'}', s_from)
        opninion_key = line[s_from+4:s_to]

        ref_opinion.set_tag(opninion_key)

        return ref_opinion

    @staticmethod
    def __parse_object(line):
        assert(isinstance(line, unicode))
        line = line[len(ContextsReader.OBJ_KEY):]

        o_begin = line.index(u"'", 0)
        o_end = line.index(u"'", o_begin + 1)

        b_from = line.index(u'b:(')
        b_to = line.index(u')', b_from)

        term_index, length = line[b_from+3:b_to].split(u',')
        terms = line[o_begin+1:o_end].split(u',')

        text_object = TextObject(terms=terms, position=int(term_index))

        sg_from = line.index(u'si:{')
        sg_to = line.index(u'}', sg_from)
        group_index = int(line[sg_from+4:sg_to])

        text_object.set_tag(group_index)

        return text_object

    @staticmethod
    def __parse_sentence(line, is_title, stemmer):
        assert(isinstance(is_title, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        key = ContextsReader.STEXT_KEY if not is_title else ContextsReader.TITLE_KEY
        text = line[len(key):]
        text = text.strip()
        return TextParser.from_string(str=text, stemmer=stemmer)

    @staticmethod
    def __parse_terms_in_title_count(line):
        line = line[len(ContextsReader.TERMS_IN_TITLE):]
        return int(line)

    @staticmethod
    def __parse_terms_in_text_count(line):
        line = line[len(ContextsReader.TERMS_IN_TEXT):]
        return int(line)

    @staticmethod
    def __parse_sentence_index(line):
        line = line[len(ContextsReader.SINDEX_KEY):]
        return int(line)

    @staticmethod
    def __parse_text_index(line):
        line = line[len(ContextsReader.TEXT_IND_KEY):]
        return int(line)
