# -*- coding: utf-8 -*-
from arekit.common import utils
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news import News
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.synonyms import SynonymsCollection

from arekit.processing.text.parser import TextParser

from arekit.source.rusentrel.context.collection import RuSentRelTextOpinionCollection
from arekit.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions
from arekit.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelNews(News):

    def __init__(self, doc_id, sentences, entities):
        assert(isinstance(sentences, list))
        assert(isinstance(entities, RuSentRelDocumentEntityCollection))

        super(RuSentRelNews, self).__init__(news_id=doc_id)

        self.__sentences = sentences
        self.__entities = entities

    # region properties

    @property
    def SentencesCount(self):
        return len(self.__sentences)

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
        return self.__sentences[index]

    def iter_sentences(self):
        for sentence in self.__sentences:
            yield sentence

    # endregion

    # region base News

    def iter_wrapped_linked_text_opinions(self, opinions):
        assert(isinstance(opinions, OpinionCollection))
        for entries in self.__iter_rusentrel_text_opinion_entries(opinions=opinions):
            yield LinkedTextOpinionsWrapper(linked_text_opinions=[entry.to_text_opinion() for entry in entries])

    def _parse_core(self, options):
        assert(isinstance(options, RuSentRelNewsParseOptions))
        parsed_sentences_iter = self.__iter_parsed_sentences(options)
        return ParsedNews(news_id=self.ID,
                          parsed_sentences=parsed_sentences_iter)

    # region private methods

    def __iter_parsed_sentences(self, options):
        assert(isinstance(options, RuSentRelNewsParseOptions))

        for s_index, sentence in enumerate(self.iter_sentences()):

            string_iter = utils.iter_text_with_substitutions(
                text=sentence.Text,
                iter_subs=sentence.iter_entity_with_local_bounds())

            yield TextParser.parse_string_list(string_iter=string_iter,
                                               # TODO. Tokens hiding actually discarded
                                               keep_tokens=options.KeepTokens,
                                               stemmer=options.Stemmer)

    def __iter_rusentrel_text_opinion_entries(self, opinions):
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
