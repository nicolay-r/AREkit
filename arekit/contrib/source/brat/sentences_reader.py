from arekit.common.entities.collection import EntityCollection
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.sentence import BratSentence


class BratDocumentSentencesReader(object):

    @staticmethod
    def from_file(input_file, entities, line_handler=None, skip_entity_func=None):
        assert(isinstance(entities, EntityCollection))
        assert(callable(skip_entity_func) or skip_entity_func is None)

        sentences_data = BratDocumentSentencesReader._parse_sentences(
            input_file=input_file, line_handler=line_handler)

        sentence_entities = BratDocumentSentencesReader._parse_entities(
            sentences_data=sentences_data,
            entities=entities,
            skip_entity_func=skip_entity_func)

        # Convert all the content to brat sentences.
        brat_sentences = []
        for s_ind, s_dict in enumerate(sentences_data):
            brat_sentence = BratSentence(text=s_dict["text"],
                                         index_begin=s_dict["ind_begin"],
                                         entities=sentence_entities[s_ind])
            brat_sentences.append(brat_sentence)

        return brat_sentences

    @staticmethod
    def from_sentences_data(entities, sentences_data, skip_entity_func=None):
        assert(isinstance(entities, EntityCollection))

        sentence_entities = BratDocumentSentencesReader._parse_entities(
            sentences_data=sentences_data,
            entities=entities,
            skip_entity_func=skip_entity_func)

        # Convert all the content to brat sentences.
        brat_sentences = []
        for s_ind, s_dict in enumerate(sentences_data):
            brat_sentence = BratSentence(text=s_dict["text"],
                                         index_begin=s_dict["ind_begin"],
                                         entities=sentence_entities[s_ind])
            brat_sentences.append(brat_sentence)

        return brat_sentences

    @staticmethod
    def __is_sentence_contains(sentence_data, entity):
        assert(isinstance(sentence_data, dict))
        assert(isinstance(entity, BratEntity))
        return entity.IndexBegin >= sentence_data["ind_begin"] and \
               entity.IndexEnd <= sentence_data["ind_end"]

    @staticmethod
    def _parse_entities(sentences_data, entities, skip_entity_func):
        """ Sentences is a list of json-like data (dictionaries).
        """
        assert(isinstance(sentences_data, list))
        assert(isinstance(entities, EntityCollection))

        entities_in_sentences = [[] for _ in range(len(sentences_data))]

        s_ind = 0
        e_ind = 0

        while s_ind < len(sentences_data) and e_ind < len(entities):
            e = entities.get_entity_by_index(e_ind)
            assert (isinstance(e, BratEntity))

            s = sentences_data[s_ind]
            entities_in_sentence = entities_in_sentences[s_ind]

            # If entity goes after the current sentence.
            if e.IndexBegin > s["ind_end"]:
                s_ind += 1
                continue

            if skip_entity_func is not None and skip_entity_func(e):
                e_ind += 1
                continue

            if BratDocumentSentencesReader.__is_sentence_contains(sentence_data=s, entity=e):
                entities_in_sentence.append(e)
                e_ind += 1
                continue

            if e.IndexEnd > s["ind_end"]:
                # Intersects with the right border of sentence
                s_ind += 1
                continue

            if e.IndexBegin < s["ind_begin"]:
                # Intersects with the left border of sentence
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}, s_b: [{} {}]".format(
                e_ind,
                e.Value, e.IndexBegin, e.IndexEnd,
                s_ind,
                s["ind_begin"], s["ind_end"]))

        return entities_in_sentences

    @staticmethod
    def _parse_sentences(input_file, line_handler):
        assert(callable(line_handler) or line_handler is None)
        sentences = []
        line_start = 0

        for line in input_file.readlines():

            line = line.decode('utf-8')
            handled_line = line_handler(line) if line_handler is not None else line

            assert(len(line) == len(handled_line))

            line_end = line_start + len(handled_line) - 1

            if handled_line != str('\r\n'):
                sentences.append({"text": handled_line, "ind_begin": line_start, "ind_end": line_end})

            line_start = line_end + 1

        return sentences
