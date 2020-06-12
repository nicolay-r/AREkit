#!/usr/bin/python
import sys
import logging
import unittest
from pymystem3 import Mystem

sys.path.append('../')


from arekit.common.entities.base import Entity
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.entities.types import EntityType
from arekit.common.parsed_news.term_position import TermPositionTypes
from arekit.contrib.bert.entity.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.token import Token
from arekit.source.rusentrel.news.base import RuSentRelNews
from arekit.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection


class TestTextOpinionsInRuSentrel(unittest.TestCase):

    @staticmethod
    def __terms_to_str(terms):
        r = []
        for t in terms:
            if isinstance(t, Token):
                r.append(t.get_token_value())
            elif isinstance(t, Entity):
                r.append(u"[{}]".format(t.Value))
            else:
                r.append(t)
        return r

    @staticmethod
    def __process(terms, entities_formatter, s_ind, t_ind):
        assert(isinstance(entities_formatter, RussianEntitiesCasedFormatter))

        r = []
        for i, term in enumerate(terms):
            result = None
            if isinstance(term, Entity):
                if i == s_ind:
                    result = entities_formatter.to_string(term, EntityType.Subject)
                if i == t_ind:
                    result = entities_formatter.to_string(term, EntityType.Object)
                else:
                    result = entities_formatter.to_string(term, EntityType.Other)
            elif isinstance(term, Token):
                result = term.get_token_value()
            else:
                result = term
            r.append(result)
        return r

    @staticmethod
    def __is_same_sentence(text_opinion, parsed_news):
        assert(isinstance(text_opinion, TextOpinion))
        s_ind = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                position_type=TermPositionTypes.SentenceIndex)
        t_ind = parsed_news.get_entity_position(id_in_document=text_opinion.TargetId,
                                                position_type=TermPositionTypes.SentenceIndex)
        return s_ind == t_ind

    def test_opinions_in_rusentrel(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logging.basicConfig(level=logging.DEBUG)
        stemmer = MystemWrapper()
        synonyms = RuSentRelSynonymsCollection.load_collection(stemmer=stemmer)

        doc_id = 47

        logger.info(u"NewsID: {}".format(doc_id))

        opinions = RuSentRelOpinionCollection.load_collection(doc_id,
                                                              synonyms=synonyms)

        assert(isinstance(opinions, OpinionCollection))

        news = RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=synonyms)

        options = RuSentRelNewsParseOptions(stemmer=stemmer,
                                            keep_tokens=True)
        parsed_news = news.parse(options,
                                 frame_variant_collection=None)
        assert(isinstance(parsed_news, ParsedNews))

        mw = POSMystemWrapper(mystem=Mystem(entire_input=False))

        entities_formatter = RussianEntitiesCasedFormatter(
            pos_tagger=POSMystemWrapper(Mystem(entire_input=False)))

        for wrap in news.iter_wrapped_linked_text_opinions(opinions):

            if len(wrap) == 0:
                continue

            text_opinion = wrap.First
            assert(isinstance(text_opinion, TextOpinion))
            text_opinion.set_owner(opinions)
            assert(isinstance(wrap, LinkedTextOpinionsWrapper))

            is_same_sentence = self.__is_same_sentence(text_opinion=text_opinion,
                                                       parsed_news=parsed_news)

            if not is_same_sentence:
                continue

            s_index = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                      position_type=TermPositionTypes.SentenceIndex)

            terms = list(parsed_news.iter_sentence_terms(s_index, return_id=False))
            str_terms = self.__terms_to_str(terms)
            str_terms_joined = u" ".join(str_terms).strip()

            s_ind = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                    position_type=TermPositionTypes.IndexInSentence)
            t_ind = parsed_news.get_entity_position(id_in_document=text_opinion.TargetId,
                                                    position_type=TermPositionTypes.IndexInSentence)

            logger.info(str_terms_joined)
            logger.info("text_opinion: {}->{}".format(text_opinion.SourceId, text_opinion.TargetId))
            logger.info("[{}] {}->{}".format(s_index, s_ind, t_ind))
            logger.info("'{}' -> '{}'".format(terms[s_ind], terms[t_ind]))
            logger.info(str_terms_joined)
            logger.info(u" ".join(self.__process(terms=terms,
                                                 entities_formatter=entities_formatter,
                                                 s_ind=s_ind,
                                                 t_ind=t_ind)))

            self.assert_(isinstance(terms[s_ind], Entity))
            self.assert_(isinstance(terms[t_ind], Entity))


if __name__ == '__main__':
    unittest.main()
