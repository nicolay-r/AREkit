from arekit.common.entities.collection import EntityCollection
from arekit.common.news.base import News
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.sentence import BratSentence


class BratDocumentReader(object):

    # TODO. Transform into sentences reader
    # TODO. Return list of sentences.
    @staticmethod
    def from_file(doc_id, input_file, entities):
        assert(isinstance(doc_id, int))
        assert(isinstance(entities, EntityCollection))

        sentences = BratDocumentReader.__read_sentences(input_file)

        s_ind = 0
        e_ind = 0

        while s_ind < len(sentences) and e_ind < len(entities):
            e = entities.get_entity_by_index(e_ind)
            assert(isinstance(e, BratEntity))

            s = sentences[s_ind]

            if s.is_entity_goes_after(e):
                s_ind += 1
                continue

            if e in s:
                s.add_local_entity(entity=e)
                e_ind += 1
                continue

            if e.Value in ['author', 'unknown']:
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}".format(
                e_ind,
                e.Value.encode('utf-8'), e.CharIndexBegin, e.CharIndexEnd,
                s_ind))

        assert(e_ind == len(entities))

        return News(doc_id=doc_id, sentences=sentences)

    # endregion

    # region private methods

    @staticmethod
    def __read_sentences(input_file):
        sentences = []
        line_start = 0

        # TODO. #319 Consider it as a special case for RuSentRel.
        unknown_entity = "Unknown}"

        for line in input_file.readlines():

            line = line.decode('utf-8')

            # TODO. #319 Consider it as a special case for RuSentRel.
            if unknown_entity in line:
                offset = line.index(unknown_entity) + len(unknown_entity)
                line_start += offset
                line = line[offset:]

            line_end = line_start + len(line) - 1

            if line != str('\r\n'):
                s = BratSentence(text=line,
                                 char_ind_begin=line_start,
                                 char_ind_end=line_end)
                sentences.append(s)

            line_start = line_end + 1

        return sentences
