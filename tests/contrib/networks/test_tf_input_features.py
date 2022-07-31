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

from arekit.common.text.stemmer import Stemmer
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.entities.base import Entity
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser

from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.networks.features.term_indices import IndicesFeature
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.entities.formatters.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper


class RuSentRelSynonymsCollectionProvider(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))
        return StemmerBasedSynonymCollection(
            iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(version),
            debug=debug,
            stemmer=stemmer,
            is_read_only=is_read_only)


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

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True),
                                               LemmasBasedFrameVariantsParser(
                                                   frame_variants=self.unique_frame_variants,
                                                   stemmer=self.stemmer,
                                                   save_lemmas=True),
                                               FrameVariantsSentimentNegation()])

        random.seed(10)
        for doc_id in [35, 36]: # RuSentRelIOUtils.iter_collection_indices():

            logger.info("NewsID: {}".format(doc_id))

            news, parsed_news, opinions = init_rusentrel_doc(
                doc_id=doc_id,
                text_parser=text_parser,
                synonyms=self.synonyms)

            # Initialize service providers.
            pairs_provider = TextOpinionPairsProvider(value_to_group_id_func=self.synonyms.get_synonym_group_index)
            entity_service = EntityServiceProvider(entity_index_func=lambda brat_entity: brat_entity.ID)

            # Setup parsed news.
            pairs_provider.init_parsed_news(parsed_news)
            entity_service.init_parsed_news(parsed_news)

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
