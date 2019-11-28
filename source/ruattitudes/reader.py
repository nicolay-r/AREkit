# -*- coding: utf-8 -*-
from core.common.text_object import TextObject
from core.common.ref_opinon import RefOpinion
from core.common.labels.base import Label
from core.processing.lemmatization.base import Stemmer
from core.processing.text.parser import TextParser
from core.source.ruattitudes.io_utils import RuAttitudesIOUtils
from core.source.ruattitudes.news import RuAttitudesNews
from core.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesFormatReader(object):

    NEWS_SEP_KEY = u'--------'
    FILE_KEY = u"File:"
    OBJ_KEY = u"Object:"
    TITLE_KEY = u"Title:"
    SINDEX_KEY = u"Sentence:"
    OPINION_KEY = u"Attitude:"
    STEXT_KEY = u"Text:"
    TEXT_IND_KEY = u"TextID:"
    TERMS_IN_TITLE = u"TermsInTitle:"
    TERMS_IN_TEXT = u"TermsInText:"
    FRAMEVAR_TITLE = u"FrameVariant:"

    def __iter__(self):
        pass

    @staticmethod
    def iter_news(stemmer=None):

        it = RuAttitudesIOUtils.iter_from_zip(
            inner_path=RuAttitudesIOUtils.get_collection_filepath(),
            process_func=lambda input_file: RuAttitudesFormatReader.__iter_news(input_file, stemmer))

        for news in it:
            yield news

    # region private methods

    @staticmethod
    def __iter_news(input_file, stemmer=None):
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        reset = False
        title = None
        title_terms_count = None
        text_terms_count = None
        sentences = []
        opinions_list = []
        objects_list = []
        s_index = 0
        news_index = None

        for line in input_file.readlines():
            line = line.decode('utf-8')

            if RuAttitudesFormatReader.FILE_KEY in line:
                pass

            if RuAttitudesFormatReader.OBJ_KEY in line:
                object = RuAttitudesFormatReader.__parse_object(line)
                objects_list.append(object)

            if RuAttitudesFormatReader.OPINION_KEY in line:
                opinion = RuAttitudesFormatReader.__parse_opinion(line, objects_list=objects_list)
                opinions_list.append(opinion)

            if RuAttitudesFormatReader.FRAMEVAR_TITLE in line:
                # TODO. This information is ommited now.
                pass

            if RuAttitudesFormatReader.TERMS_IN_TITLE in line:
                title_terms_count = RuAttitudesFormatReader.__parse_terms_in_title_count(line)

            if RuAttitudesFormatReader.SINDEX_KEY in line:
                s_index = RuAttitudesFormatReader.__parse_sentence_index(line)

            if RuAttitudesFormatReader.TEXT_IND_KEY in line:
                news_index = RuAttitudesFormatReader.__parse_text_index(line)

            if RuAttitudesFormatReader.TITLE_KEY in line:
                title = RuAttitudesSentence(is_title=True,
                                            parsed_text=RuAttitudesFormatReader.__parse_sentence(line,
                                                                                                 is_title=True,
                                                                                                 stemmer=stemmer),
                                            ref_opinions=opinions_list,
                                            objects_list=objects_list,
                                            sentence_index=-1)
                sentences.append(title)
                assert(title_terms_count == len(title.ParsedText) or title_terms_count is None)
                reset = True

            if RuAttitudesFormatReader.STEXT_KEY in line and line.index(RuAttitudesFormatReader.STEXT_KEY) == 0:
                sentence = RuAttitudesSentence(is_title=False,
                                               parsed_text=RuAttitudesFormatReader.__parse_sentence(line,
                                                                                                    is_title=False,
                                                                                                    stemmer=stemmer),
                                               ref_opinions=opinions_list,
                                               objects_list=objects_list,
                                               sentence_index=s_index)
                sentences.append(sentence)
                assert(text_terms_count == len(sentence.ParsedText) or text_terms_count is None)
                reset = True

            if RuAttitudesFormatReader.NEWS_SEP_KEY in line and title is not None:
                yield RuAttitudesNews(sentences=sentences,
                                      news_index=news_index)
                sentences = []
                reset = True

            if RuAttitudesFormatReader.TERMS_IN_TEXT in line:
                text_terms_count = RuAttitudesFormatReader.__parse_terms_in_text_count(line)

            if reset:
                opinions_list = []
                objects_list = []
                title_terms_count = None
                reset = False

        if len(sentences) > 0:
            yield RuAttitudesNews(sentences=sentences,
                                  news_index=news_index)
            sentences = []

        assert(len(sentences) == 0)

    @staticmethod
    def __parse_opinion(line, objects_list):
        assert(isinstance(objects_list, list))

        line = line[len(RuAttitudesFormatReader.OPINION_KEY):]

        s_from = line.index(u'b:(')
        s_to = line.index(u')', s_from)
        label = Label.from_int(int(line[s_from+3:s_to]))

        o_from = line.index(u'oi:[')
        o_to = line.index(u']', o_from)
        source_object_id, target_object_id = line[o_from + 4:o_to].split(u',')

        source_object_id = int(source_object_id)
        target_object_id = int(target_object_id)

        ref_opinion = RefOpinion(source_id=source_object_id,
                                 target_id=target_object_id,
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
        line = line[len(RuAttitudesFormatReader.OBJ_KEY):]

        o_begin = line.index(u"'", 0)
        o_end = line.index(u"'", o_begin + 1)

        b_from = line.index(u'b:(')
        b_to = line.index(u')', b_from)

        term_index, length = line[b_from+3:b_to].split(u',')
        terms = line[o_begin+1:o_end].split(u',')

        obj_type = RuAttitudesFormatReader.__try_get_type(line)

        text_object = TextObject(terms=terms,
                                 obj_type=obj_type,
                                 position=int(term_index))

        sg_from = line.index(u'si:{')
        sg_to = line.index(u'}', sg_from)
        group_index = int(line[sg_from+4:sg_to])

        text_object.set_tag(group_index)

        return text_object

    @staticmethod
    def __parse_sentence(line, is_title, stemmer):
        assert(isinstance(is_title, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        key = RuAttitudesFormatReader.STEXT_KEY if not is_title else RuAttitudesFormatReader.TITLE_KEY
        text = line[len(key):]
        text = text.strip()
        return TextParser.from_string(str=text, stemmer=stemmer)

    @staticmethod
    def __parse_terms_in_title_count(line):
        line = line[len(RuAttitudesFormatReader.TERMS_IN_TITLE):]
        return int(line)

    @staticmethod
    def __parse_terms_in_text_count(line):
        line = line[len(RuAttitudesFormatReader.TERMS_IN_TEXT):]
        return int(line)

    @staticmethod
    def __parse_sentence_index(line):
        line = line[len(RuAttitudesFormatReader.SINDEX_KEY):]
        return int(line)

    @staticmethod
    def __parse_text_index(line):
        line = line[len(RuAttitudesFormatReader.TEXT_IND_KEY):]
        return int(line)

    @staticmethod
    def __try_get_type(line):
        template = u't:['

        if template not in line:
            return None

        t_from = line.index(template)
        t_to = line.index(u']', t_from)
        return line[t_from + len(template):t_to]

    # endregion
