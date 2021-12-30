import logging
import random
import sys
import unittest

import numpy as np
from pymystem3 import Mystem

sys.path.append('../../../')

from tests.contrib.networks.text.news import init_rusentrel_doc
from tests.text.linked_opinions import iter_same_sentence_linked_text_opinions
from tests.text.utils import terms_to_str

from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.entities.base import Entity
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser

from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.networks.features.term_indices import IndicesFeature
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.experiment_rusentrel.entities.str_rus_cased_fmt import RussianEntitiesCasedFormatter

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser


class TestTfInputFeatures(unittest.TestCase):

    X_PAD_VALUE = 0

    @classmethod
    def setUpClass(cls):
        cls.stemmer = MystemWrapper()
        cls.entities_formatter = RussianEntitiesCasedFormatter(
            pos_tagger=POSMystemWrapper(Mystem(entire_input=False)))
        cls.synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=cls.stemmer)
        cls.frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V10)

        cls.unique_frame_variants = FrameVariantsCollection()
        cls.unique_frame_variants.fill_from_iterable(
            variants_with_id=cls.frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

    def test(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        text_parser = BaseTextParser(pipeline=[
            RuSentRelTextEntitiesParser(),
            DefaultTextTokenizer(keep_tokens=True),
            LemmasBasedFrameVariantsParser(frame_variants=self.unique_frame_variants,
                                           stemmer=self.stemmer,
                                           save_lemmas=True)
        ])

        random.seed(10)
        for doc_id in [35, 36]: # RuSentRelIOUtils.iter_collection_indices():

            logger.info("NewsID: {}".format(doc_id))

            news, parsed_news, opinions = init_rusentrel_doc(
                doc_id=doc_id,
                text_parser=text_parser,
                synonyms=self.synonyms)

            pairs_provider = TextOpinionPairsProvider(parsed_news=parsed_news,
                                                      value_to_group_id_func=self.synonyms.get_synonym_group_index)
            entity_service = EntityServiceProvider(parsed_news=parsed_news)

            text_opinion_iter = iter_same_sentence_linked_text_opinions(pairs_provider=pairs_provider,
                                                                        entity_service=entity_service,
                                                                        opinions=opinions)

            for text_opinion in text_opinion_iter:

                s_index = entity_service.get_entity_position(id_in_document=text_opinion.SourceId,
                                                             position_type=TermPositionTypes.SentenceIndex)

                s_ind = entity_service.get_entity_position(id_in_document=text_opinion.SourceId,
                                                           position_type=TermPositionTypes.IndexInSentence)
                t_ind = entity_service.get_entity_position(id_in_document=text_opinion.TargetId,
                                                           position_type=TermPositionTypes.IndexInSentence)

                terms = list(parsed_news.iter_sentence_terms(s_index, return_id=False))
                x_feature = IndicesFeature.from_vector_to_be_fitted(
                    value_vector=np.array(terms),
                    e1_ind=s_ind,
                    e2_ind=t_ind,
                    expected_size=random.randint(50, 60),
                    filler="<PAD>")

                cropped_terms = x_feature.ValueVector
                subj_ind = s_ind - x_feature.StartIndex
                obj_ind = t_ind - x_feature.StartIndex

                logger.info(len(terms))
                logger.info("Source Index: {}".format(subj_ind))
                logger.info("Target Index: {}".format(obj_ind))
                s = " ".join(terms_to_str(cropped_terms))
                logger.info("Result: {}".format(s))

                assert(isinstance(x_feature.ValueVector[subj_ind], Entity))
                assert(isinstance(x_feature.ValueVector[obj_ind], Entity))


if __name__ == '__main__':
    unittest.main()
