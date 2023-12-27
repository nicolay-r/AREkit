import logging
import random
import sys
import unittest

import numpy as np


sys.path.append('../')

from arekit.common.text.stemmer import Stemmer
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.docs.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.entities.base import Entity
from arekit.common.docs.parsed.term_position import TermPositionTypes
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter
from doc import init_rusentrel_doc
from indices_feature import IndicesFeature
from labels import TestPositiveLabel, TestNegativeLabel
from utils import iter_same_sentence_linked_text_opinions, terms_to_str


class RuSentRelSynonymsCollectionProvider(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))
        return StemmerBasedSynonymCollection(
            iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(version),
            stemmer=stemmer,
            is_read_only=is_read_only)


class TestTfInputFeatures(unittest.TestCase):

    X_PAD_VALUE = 0

    @classmethod
    def setUpClass(cls):
        cls.stemmer = MystemWrapper()
        cls.entities_formatter = StringEntitiesDisplayValueFormatter()
        cls.synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=cls.stemmer)
        cls.frames_collection = RuSentiFramesCollection.read(
            version=RuSentiFramesVersions.V10,
            labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=TestPositiveLabel,
                                                    neg_label_type=TestNegativeLabel),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=TestPositiveLabel,
                                                                 neg_label_type=TestNegativeLabel))

        cls.unique_frame_variants = FrameVariantsCollection()
        cls.unique_frame_variants.fill_from_iterable(
            variants_with_id=cls.frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

    def test(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(src_key="input"),
                                               DefaultTextTokenizer(keep_tokens=True),
                                               LemmasBasedFrameVariantsParser(
                                                   frame_variants=self.unique_frame_variants,
                                                   stemmer=self.stemmer,
                                                   save_lemmas=True),
                                               FrameVariantsSentimentNegation()])

        random.seed(10)
        for doc_id in [35, 36]: # RuSentRelIOUtils.iter_collection_indices():

            logger.info("DocumentID: {}".format(doc_id))

            doc, parsed_doc, opinions = init_rusentrel_doc(
                doc_id=doc_id, text_parser=text_parser, synonyms=self.synonyms)

            # Initialize service providers.
            pairs_provider = TextOpinionPairsProvider(value_to_group_id_func=self.synonyms.get_synonym_group_index)
            entity_service = EntityServiceProvider(entity_index_func=lambda brat_entity: brat_entity.ID)

            # Setup parsed doc.
            pairs_provider.init_parsed_doc(parsed_doc)
            entity_service.init_parsed_doc(parsed_doc)

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

                terms = list(parsed_doc.iter_sentence_terms(s_index, return_id=False))

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
