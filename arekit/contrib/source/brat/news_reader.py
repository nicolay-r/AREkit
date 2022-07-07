from arekit.common.entities.collection import EntityCollection
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.sentence import BratSentence


class BratDocumentSentencesReader(object):

    @staticmethod
    def from_file(input_file, entities, line_handler=None, skip_entity_func=None):
        assert (isinstance(entities, EntityCollection))
        assert (callable(skip_entity_func) or skip_entity_func is None)

        sentences = BratDocumentSentencesReader.__parse_sentences(input_file=input_file,
                                                                  line_handler=line_handler)

        s_ind = 0
        e_ind = 0

        while s_ind < len(sentences) and e_ind < len(entities):
            e = entities.get_entity_by_index(e_ind)
            assert (isinstance(e, BratEntity))

            s = sentences[s_ind]

            if s.is_entity_goes_after(e):
                s_ind += 1
                continue

            if e in s:
                s.add_local_entity(entity=e)
                e_ind += 1
                continue

            if skip_entity_func is not None and skip_entity_func(e):
                e_ind += 1
                continue

            if e.CharIndexEnd > s.EndBound:
                # Intersects with the right border of sentence
                s_ind += 1
                continue

            if e.CharIndexBegin < s.BeginBound:
                # Intersects with the left border of sentence
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}, s_b: [{} {}]".format(
                e_ind,
                e.Value, e.CharIndexBegin, e.CharIndexEnd,
                s_ind,
                s.BeginBound, s.EndBound))

        return sentences

    # endregion

    # region private methods

    @staticmethod
    def __parse_sentences(input_file, line_handler):
        assert(callable(line_handler) or line_handler is None)
        sentences = []
        line_start = 0

        for line in input_file.readlines():

            line = line.decode('utf-8')
            handled_line = line_handler(line) if line_handler is not None else line

            assert(len(line) == len(handled_line))

            line_end = line_start + len(handled_line) - 1

            if handled_line != str('\r\n'):
                s = BratSentence(text=handled_line,
                                 char_ind_begin=line_start,
                                 char_ind_end=line_end)
                sentences.append(s)

            line_start = line_end + 1

        return sentences