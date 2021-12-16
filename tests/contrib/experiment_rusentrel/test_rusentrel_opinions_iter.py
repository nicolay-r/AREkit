#!/usr/bin/python
import sys
import logging
import unittest
from pymystem3 import Mystem

sys.path.append('../../../../')

from tests.text.utils import terms_to_str
from tests.text.linked_opinions import iter_same_sentence_linked_text_opinions

from tests.contrib.source.text.news import init_rusentrel_doc
from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.experiment_rusentrel.entities.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.contrib.experiment_rusentrel.frame_variants import ExperimentFrameVariantsCollection
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import \
    ExperimentRuSentiFramesLabelsFormatter, \
    ExperimentRuSentiFramesEffectLabelsFormatter

from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.entities.base import Entity
from arekit.common.entities.types import EntityType
from arekit.common.text.parser import BaseTextParser
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.token import Token
from arekit.processing.text.pipeline_frames import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer


class TestRuSentRelOpinionsIter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.entities_formatter = RussianEntitiesCasedFormatter(
            pos_tagger=POSMystemWrapper(Mystem(entire_input=False)))
        cls.stemmer = MystemWrapper()
        cls.synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=cls.stemmer)
        cls.frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V10,
            labels_fmt=ExperimentRuSentiFramesLabelsFormatter(),
            effect_labels_fmt=ExperimentRuSentiFramesEffectLabelsFormatter())

        cls.unique_frame_variants = ExperimentFrameVariantsCollection(stemmer=cls.stemmer)
        cls.unique_frame_variants.fill_from_iterable(
            variants_with_id=cls.frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

    @staticmethod
    def __process(terms, entities_formatter, s_ind, t_ind):
        assert(isinstance(entities_formatter, StringEntitiesFormatter))

        r = []
        for i, term in enumerate(terms):
            if isinstance(term, Entity):
                if i == s_ind:
                    result = entities_formatter.to_string(term, EntityType.Subject)
                elif i == t_ind:
                    result = entities_formatter.to_string(term, EntityType.Object)
                else:
                    result = entities_formatter.to_string(term, EntityType.Other)
            elif isinstance(term, Token):
                result = term.get_token_value()
            else:
                result = term
            r.append(result)
        return r

    def test_opinions_in_rusentrel(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        # Initialize text parser pipeline.
        text_parser = BaseTextParser(pipeline=[
            RuSentRelTextEntitiesParser(),
            DefaultTextTokenizer(keep_tokens=True),
            LemmasBasedFrameVariantsParser(frame_variants=self.unique_frame_variants,
                                           stemmer=self.stemmer)
        ])

        # Initialize specific document
        doc_id = 47
        logger.info("NewsID: {}".format(doc_id))
        news, parsed_news, opinions = init_rusentrel_doc(
            doc_id=doc_id,
            text_parser=text_parser,
            synonyms=self.synonyms)

        for text_opinion in iter_same_sentence_linked_text_opinions(news=news,
                                                                    opinions=opinions,
                                                                    parsed_news=parsed_news):

            s_index = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                      position_type=TermPositionTypes.SentenceIndex)

            terms = list(parsed_news.iter_sentence_terms(s_index, return_id=False))
            str_terms_joined = " ".join(terms_to_str(terms)).strip()

            s_ind = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                    position_type=TermPositionTypes.IndexInSentence)
            t_ind = parsed_news.get_entity_position(id_in_document=text_opinion.TargetId,
                                                    position_type=TermPositionTypes.IndexInSentence)

            logger.info("text_opinion: {}->{}".format(text_opinion.SourceId, text_opinion.TargetId))
            logger.info("[{}] {}->{}".format(s_index, s_ind, t_ind))
            logger.info("'{}' -> '{}'".format(terms[s_ind], terms[t_ind]))
            logger.info(str_terms_joined)

            logger.info(" ".join(self.__process(terms=terms,
                                                entities_formatter=self.entities_formatter,
                                                s_ind=s_ind,
                                                t_ind=t_ind)))

            self.assertTrue(isinstance(terms[s_ind], Entity))
            self.assertTrue(isinstance(terms[t_ind], Entity))


if __name__ == '__main__':
    unittest.main()
