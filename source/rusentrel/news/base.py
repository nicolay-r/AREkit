# -*- coding: utf-8 -*-
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news import News
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection

from arekit.source.rusentrel.context.collection import RuSentRelTextOpinionCollection
from arekit.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelNews(News):

    def __init__(self, doc_id, sentences, entities):
        assert(isinstance(sentences, list))
        assert(isinstance(entities, RuSentRelDocumentEntityCollection))

        super(RuSentRelNews, self).__init__(news_id=doc_id,
                                            sentences=sentences,
                                            entities_parser=RuSentRelTextEntitiesParser())

        self.__entities = entities

    # region properties

    @property
    def SentencesCount(self):
        return len(self._sentences)

    @property
    def DocEntities(self):
        return self.__entities

    # endregion

    # region class methods

    @classmethod
    def read_document(cls, doc_id, synonyms, version=RuSentRelVersions.V11):
        assert(isinstance(synonyms, SynonymsCollection))

        entities = RuSentRelDocumentEntityCollection.read_collection(
            doc_id=doc_id,
            synonyms=synonyms,
            version=version)

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_news_innerpath(doc_id),
            process_func=lambda input_file: cls.__from_file(doc_id=doc_id,
                                                            input_file=input_file,
                                                            entities=entities),
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

            if e.Value in [u'author', u'unknown']:
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

    # endregion

    # region public methods

    def get_sentence_by_index(self, index):
        return self._sentences[index]

    # endregion

    # region base News

    def iter_wrapped_linked_text_opinions(self, opinions):
        assert(isinstance(opinions, OpinionCollection))
        for text_opinions in self.__iter_rusentrel_text_opinions(opinions=opinions):
            yield LinkedTextOpinionsWrapper(linked_text_opinions=[text_opinion for text_opinion in text_opinions])

    # region private methods

    def __iter_rusentrel_text_opinions(self, opinions):
        """
        Document Level Opinions -> Linked Text Level Opinions
        """
        assert(isinstance(opinions, OpinionCollection))

        for opinion in opinions:
            yield RuSentRelTextOpinionCollection.from_opinions(rusentrel_news_id=self.ID,
                                                               doc_entities=self.DocEntities,
                                                               opinions=[opinion])

    # endregion

    # endregion
