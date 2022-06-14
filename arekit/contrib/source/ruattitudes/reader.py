from arekit.common.utils import split_by_whitespaces
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence
from arekit.contrib.source.ruattitudes.text_object import TextObject


class RuAttitudesFormatReader(object):

    DOC_SEP_KEY = '--------'
    FILE_KEY = "File:"
    OBJ_KEY = "Object:"
    TITLE_KEY = "Title:"
    SINDEX_KEY = "Sentence:"
    OPINION_KEY = "Attitude:"
    STEXT_KEY = "Text:"
    TERMS_IN_TITLE = "TermsInTitle:"
    TERMS_IN_TEXT = "TermsInText:"
    FRAMEVAR_TITLE = "FrameVariant:"

    AUTH_LABEL = '<AUTH>'

    def __iter__(self):
        pass

    # region private methods

    @staticmethod
    def iter_news_inds(input_file, get_news_index_func):
        assert(callable(get_news_index_func))

        title = None
        local_news_ind = 0
        has_sentences = False

        for line in RuAttitudesFormatReader.__iter_lines(input_file):

            if RuAttitudesFormatReader.__check_is_title(line):
                # We use a placeholder, there is no need in actual value out there.
                title = "title"
                has_sentences = True

            if RuAttitudesFormatReader.__check_is_news_sep(line=line, title=title):
                yield RuAttitudesFormatReader.__assign_news_index(news_index_func=get_news_index_func,
                                                                  local_index=local_news_ind)
                local_news_ind += 1
                title = None

        if has_sentences:
            yield RuAttitudesFormatReader.__assign_news_index(news_index_func=get_news_index_func,
                                                              local_index=local_news_ind)

    @staticmethod
    def iter_news(input_file, get_news_index_func):
        assert(callable(get_news_index_func))

        reset = False
        title = None
        title_terms_count = None
        text_terms_count = None
        sentences = []
        opinions_list = []
        objects_list = []
        s_index = 0
        objects_in_prior_sentences_count = 0
        local_news_ind = 0

        for line in RuAttitudesFormatReader.__iter_lines(input_file):

            if RuAttitudesFormatReader.FILE_KEY in line:
                pass

            if RuAttitudesFormatReader.OBJ_KEY in line:
                object = RuAttitudesFormatReader.__parse_object(line)
                objects_list.append(object)

            if RuAttitudesFormatReader.OPINION_KEY in line:
                sentence_opin = RuAttitudesFormatReader.__parse_sentence_opin(line)
                opinions_list.append(sentence_opin)

            if RuAttitudesFormatReader.FRAMEVAR_TITLE in line:
                # TODO. This information is ommited now.
                pass

            if RuAttitudesFormatReader.TERMS_IN_TITLE in line:
                title_terms_count = RuAttitudesFormatReader.__parse_terms_in_title_count(line)

            if RuAttitudesFormatReader.SINDEX_KEY in line:
                s_index = RuAttitudesFormatReader.__parse_sentence_index(line)

            if RuAttitudesFormatReader.__check_is_title(line):
                title = RuAttitudesSentence(is_title=True,
                                            text=RuAttitudesFormatReader.__parse_sentence(line, True),
                                            sentence_opins=opinions_list,
                                            objects_list=objects_list,
                                            sentence_index=-1)
                sentences.append(title)
                t_len = RuAttitudesFormatReader.__calculate_terms_in_line(line)
                assert(title_terms_count == t_len or title_terms_count is None)
                reset = True

            if RuAttitudesFormatReader.STEXT_KEY in line and line.index(RuAttitudesFormatReader.STEXT_KEY) == 0:
                sentence = RuAttitudesSentence(is_title=False,
                                               text=RuAttitudesFormatReader.__parse_sentence(line, False),
                                               sentence_opins=opinions_list,
                                               objects_list=objects_list,
                                               sentence_index=s_index)
                sentences.append(sentence)
                objects_in_prior_sentences_count += len(objects_list)
                t_len = RuAttitudesFormatReader.__calculate_terms_in_line(line)
                assert(text_terms_count == t_len or text_terms_count is None)
                reset = True

            if RuAttitudesFormatReader.__check_is_news_sep(line=line, title=title):
                news_index = RuAttitudesFormatReader.__assign_news_index(news_index_func=get_news_index_func,
                                                                         local_index=local_news_ind)
                yield RuAttitudesNews(sentences=sentences,
                                      news_index=news_index)
                local_news_ind += 1
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
            news_index = RuAttitudesFormatReader.__assign_news_index(news_index_func=get_news_index_func,
                                                                     local_index=local_news_ind)
            yield RuAttitudesNews(sentences=sentences,
                                  news_index=news_index)
            sentences = []

        assert(len(sentences) == 0)

    @staticmethod
    def __assign_news_index(news_index_func, local_index):
        assert(callable(news_index_func))
        return news_index_func(local_index)

    @staticmethod
    def __check_is_news_sep(line, title):
        return RuAttitudesFormatReader.DOC_SEP_KEY in line and title is not None

    @staticmethod
    def __check_is_title(line):
        return RuAttitudesFormatReader.TITLE_KEY in line

    @staticmethod
    def __iter_lines(input_file):
        for line in input_file.readlines():
            yield line.decode('utf-8')

    @staticmethod
    def __calculate_terms_in_line(line):
        assert(isinstance(line, str))
        return len(split_by_whitespaces(line))

    @staticmethod
    def __parse_sentence(line, is_title):
        assert(isinstance(is_title, bool))

        key = RuAttitudesFormatReader.STEXT_KEY if not is_title else RuAttitudesFormatReader.TITLE_KEY
        text = line[len(key):]
        return text.strip()

    @staticmethod
    def __parse_sentence_opin(line):
        line = line[len(RuAttitudesFormatReader.OPINION_KEY):]

        s_from = line.index('b:(')
        s_to = line.index(')', s_from)
        label = int(line[s_from + 3:s_to])

        o_from = line.index('oi:[')
        o_to = line.index(']', o_from)
        source_object_id_in_sentence, target_object_id_in_sentence = line[o_from + 4:o_to].split(',')

        source_object_id_in_sentence = int(source_object_id_in_sentence)
        target_object_id_in_sentence = int(target_object_id_in_sentence)

        s_from = line.index('si:{')
        s_to = line.index('}', s_from)
        opninion_key = line[s_from+4:s_to]

        sentence_opin = SentenceOpinion(source_id=source_object_id_in_sentence,
                                        target_id=target_object_id_in_sentence,
                                        label_int=label,
                                        tag=opninion_key)

        return sentence_opin

    @staticmethod
    def __parse_object(line):
        assert(isinstance(line, str))

        line = line[len(RuAttitudesFormatReader.OBJ_KEY):]

        obj_ind_begin = line.index('oi:[', 0)
        obj_ind_end = line.index(']', obj_ind_begin + 1)

        o_begin = line.index("'", 0)
        o_end = line.index("'", o_begin + 1)

        b_from = line.index('b:(')
        b_to = line.index(')', b_from)

        id_in_sentence = int(line[obj_ind_begin + 4:obj_ind_end])
        term_index, length = line[b_from+3:b_to].split(',')
        value = line[o_begin + 1:o_end]

        obj_type = RuAttitudesFormatReader.__try_get_type(line)

        sg_from = line.index('si:{')
        sg_to = line.index('}', sg_from)
        group_index = int(line[sg_from+4:sg_to])

        is_auth = '<AUTH>' in line

        text_object = TextObject(id_in_sentence=id_in_sentence,
                                 value=value,
                                 obj_type=obj_type,
                                 position=int(term_index),
                                 terms_count=int(length),
                                 syn_group_index=group_index,
                                 is_auth=is_auth)

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
    def __try_get_type(line):

        # Tag, utilized in RuAttitudes-2.0 format.
        template = 'type:'
        if template in line:
            is_auth = RuAttitudesFormatReader.AUTH_LABEL in line
            t_from = line.index(template)
            t_to = line.index(RuAttitudesFormatReader.AUTH_LABEL[0], t_from) if is_auth else len(line)
            return line[t_from + len(template):t_to].strip()

        # Tag, utilized in RuAttitudes-1.* format.
        template = 't:['
        if template in line:
            t_from = line.index(template)
            t_to = line.index(']', t_from)
            return line[t_from + len(template):t_to].strip()

    # endregion
