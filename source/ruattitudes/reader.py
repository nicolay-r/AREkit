# -*- coding: utf-8 -*-
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.source.ruattitudes.text_object import TextObject
from arekit.source.ruattitudes.ref_opinion import RefOpinion
from arekit.source.ruattitudes.news.base import RuAttitudesNews
from arekit.source.ruattitudes.sentence import RuAttitudesSentence


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

    # region private methods

    @staticmethod
    def iter_news(input_file):
        reset = False
        title = None
        title_terms_count = None
        text_terms_count = None
        sentences = []
        opinions_list = []
        objects_list = []
        s_index = 0
        objects_in_prior_sentences_count = 0
        news_index = None

        label_scaler = ThreeLabelScaler()

        for line in input_file.readlines():
            line = line.decode('utf-8')

            if RuAttitudesFormatReader.FILE_KEY in line:
                pass

            if RuAttitudesFormatReader.OBJ_KEY in line:
                object = RuAttitudesFormatReader.__parse_object(line)
                objects_list.append(object)

            if RuAttitudesFormatReader.OPINION_KEY in line:
                opinion = RuAttitudesFormatReader.__parse_opinion(line=line,
                                                                  objects_list=objects_list,
                                                                  label_scaler=label_scaler)
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
                                            text=RuAttitudesFormatReader.__parse_sentence(line, True),
                                            ref_opinions=opinions_list,
                                            objects_list=objects_list,
                                            sentence_index=-1)
                sentences.append(title)
                t_len = RuAttitudesFormatReader.__calculate_terms_in_line(line)
                assert(title_terms_count == t_len or title_terms_count is None)
                reset = True

            if RuAttitudesFormatReader.STEXT_KEY in line and line.index(RuAttitudesFormatReader.STEXT_KEY) == 0:
                sentence = RuAttitudesSentence(is_title=False,
                                               text=RuAttitudesFormatReader.__parse_sentence(line, False),
                                               ref_opinions=opinions_list,
                                               objects_list=objects_list,
                                               sentence_index=s_index)
                sentences.append(sentence)
                objects_in_prior_sentences_count += len(objects_list)
                t_len = RuAttitudesFormatReader.__calculate_terms_in_line(line)
                assert(text_terms_count == t_len or text_terms_count is None)
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
    def __calculate_terms_in_line(line):
        assert(isinstance(line, unicode))
        return len(line.split(u' '))

    @staticmethod
    def __parse_sentence(line, is_title):
        assert(isinstance(is_title, bool))

        key = RuAttitudesFormatReader.STEXT_KEY if not is_title else RuAttitudesFormatReader.TITLE_KEY
        text = line[len(key):]
        return text.strip()

    @staticmethod
    def __parse_opinion(line, objects_list, label_scaler):
        assert(isinstance(objects_list, list))
        assert(isinstance(label_scaler, ThreeLabelScaler))

        line = line[len(RuAttitudesFormatReader.OPINION_KEY):]

        s_from = line.index(u'b:(')
        s_to = line.index(u')', s_from)
        label = label_scaler.int_to_label(int(line[s_from+3:s_to]))

        o_from = line.index(u'oi:[')
        o_to = line.index(u']', o_from)
        source_object_id_in_sentence, target_object_id_in_sentence = line[o_from + 4:o_to].split(u',')

        source_object_id_in_sentence = int(source_object_id_in_sentence)
        target_object_id_in_sentence = int(target_object_id_in_sentence)

        s_from = line.index(u'si:{')
        s_to = line.index(u'}', s_from)
        opninion_key = line[s_from+4:s_to]

        ref_opinion = RefOpinion(source_id=source_object_id_in_sentence,
                                 target_id=target_object_id_in_sentence,
                                 source_value=objects_list[source_object_id_in_sentence].get_value(),
                                 target_value=objects_list[target_object_id_in_sentence].get_value(),
                                 sentiment=label,
                                 tag=opninion_key)

        return ref_opinion

    @staticmethod
    def __parse_object(line):
        assert(isinstance(line, unicode))

        line = line[len(RuAttitudesFormatReader.OBJ_KEY):]

        obj_ind_begin = line.index(u'oi:[', 0)
        obj_ind_end = line.index(u']', obj_ind_begin + 1)

        o_begin = line.index(u"'", 0)
        o_end = line.index(u"'", o_begin + 1)

        b_from = line.index(u'b:(')
        b_to = line.index(u')', b_from)

        id_in_sentence = int(line[obj_ind_begin + 4:obj_ind_end])
        term_index, length = line[b_from+3:b_to].split(u',')
        terms = line[o_begin+1:o_end].split(u',')

        obj_type = RuAttitudesFormatReader.__try_get_type(line)

        text_object = TextObject(id_in_sentence=id_in_sentence,
                                 terms=terms,
                                 obj_type=obj_type,
                                 position=int(term_index))

        sg_from = line.index(u'si:{')
        sg_to = line.index(u'}', sg_from)
        group_index = int(line[sg_from+4:sg_to])

        text_object.set_tag(group_index)

        return text_object

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
