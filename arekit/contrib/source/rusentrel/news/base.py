from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.base import News
from arekit.common.opinions.base import Opinion
from arekit.common.synonyms import SynonymsCollection

from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.contrib.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.opinions.extraction import iter_text_opinions_by_doc_opinion
from arekit.contrib.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelNews(News):

    def __init__(self, doc_id, sentences, entities):
        assert(isinstance(sentences, list))
        assert(isinstance(entities, RuSentRelDocumentEntityCollection))

        super(RuSentRelNews, self).__init__(news_id=doc_id, sentences=sentences)

        self.__entities = entities

    # region properties

    @property
    def DocEntities(self):
        return self.__entities

    # endregion

    # region class methods

    @classmethod
    def read_document(cls, doc_id, synonyms, version=RuSentRelVersions.V11, target_doc_id=None):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(target_doc_id, int) or target_doc_id is None)

        def file_to_doc(input_file):
            return cls.__from_file(
                doc_id=target_doc_id if target_doc_id is not None else doc_id,
                input_file=input_file,
                entities=entities)

        entities = RuSentRelDocumentEntityCollection.read_collection(
            doc_id=doc_id,
            synonyms=synonyms,
            version=version)

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_news_innerpath(doc_id),
            process_func=file_to_doc,
            version=version)

    @classmethod
    def __from_file(cls, doc_id, input_file, entities):
        assert(isinstance(doc_id, int))
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

            if e.Value in ['author', 'unknown']:
                e_ind += 1
                continue

            raise Exception("e_i:{} e:('{}',{},{}), s_i:{}".format(
                e_ind,
                e.Value.encode('utf-8'), e.CharIndexBegin, e.CharIndexEnd,
                s_ind))

        assert(e_ind == len(entities))

        return cls(doc_id=doc_id,
                   sentences=sentences,
                   entities=entities)

    # endregion

    # region private methods

    @staticmethod
    def __read_sentences(input_file):
        sentences = []
        line_start = 0
        unknown_entity = "Unknown}"

        for line in input_file.readlines():

            line = line.decode('utf-8')

            if unknown_entity in line:
                offset = line.index(unknown_entity) + len(unknown_entity)
                line_start += offset
                line = line[offset:]

            line_end = line_start + len(line) - 1

            if line != str('\r\n'):
                s = RuSentRelSentence(text=line,
                                      char_ind_begin=line_start,
                                      char_ind_end=line_end)
                sentences.append(s)

            line_start = line_end + 1

        return sentences

    # endregion

    # region base News

    def extract_linked_text_opinions(self, opinion):
        assert(isinstance(opinion, Opinion))

        opinions_it = iter_text_opinions_by_doc_opinion(rusentrel_news_id=self.ID,
                                                        doc_entities=self.__entities,
                                                        opinion=opinion)

        return LinkedTextOpinionsWrapper(linked_text_opinions=opinions_it)

    def get_entities_collection(self):
        return self.__entities

    @staticmethod
    def _sentence_to_terms_list_core(sentence):
        with RuSentRelTextEntitiesParser() as parser:
            return parser.parse(sentence)

    # endregion
