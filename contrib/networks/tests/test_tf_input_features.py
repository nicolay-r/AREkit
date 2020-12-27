import logging
import random
import sys
import unittest

import numpy as np
from pymystem3 import Mystem


sys.path.append('../../../')

from arekit.common.entities.base import Entity
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.entities.formatters.str_rus_cased_fmt import RussianEntitiesCasedFormatter

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper

from arekit.contrib.networks.tests.text.news import init_rusentrel_doc
from arekit.contrib.networks.features.term_indices import IndicesFeature
from arekit.contrib.experiments.synonyms.provider import RuSentRelSynonymsCollectionProvider

from arekit.tests.text.linked_opinions import iter_same_sentence_linked_text_opinions
from arekit.tests.text.utils import terms_to_str

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions


class TestTfInputFeatures(unittest.TestCase):

    X_PAD_VALUE = 0

    @classmethod
    def setUpClass(cls):
        cls.stemmer = MystemWrapper()
        cls.entities_formatter = RussianEntitiesCasedFormatter(
            pos_tagger=POSMystemWrapper(Mystem(entire_input=False)))
        cls.synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=cls.stemmer)
        cls.frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V10)
        cls.unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=cls.frames_collection.iter_frame_id_and_variants(),
            stemmer=cls.stemmer)

    def test(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        random.seed(10)
        for doc_id in [35, 36]: # RuSentRelIOUtils.iter_collection_indices():

            logger.info(u"NewsID: {}".format(doc_id))

            news, parsed_news, opinions = init_rusentrel_doc(
                doc_id=doc_id,
                stemmer=self.stemmer,
                synonyms=self.synonyms,
                unique_frame_variants=self.unique_frame_variants)

            text_opinion_iter = iter_same_sentence_linked_text_opinions(
                news=news,
                parsed_news=parsed_news,
                opinions=opinions)

            for text_opinion in text_opinion_iter:

                s_index = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                          position_type=TermPositionTypes.SentenceIndex)

                terms = list(parsed_news.iter_sentence_terms(s_index, return_id=False))

                s_ind = parsed_news.get_entity_position(id_in_document=text_opinion.SourceId,
                                                        position_type=TermPositionTypes.IndexInSentence)
                t_ind = parsed_news.get_entity_position(id_in_document=text_opinion.TargetId,
                                                        position_type=TermPositionTypes.IndexInSentence)

                x_feature = IndicesFeature.from_vector_to_be_fitted(
                    value_vector=np.array(terms),
                    e1_ind=s_ind,
                    e2_ind=t_ind,
                    expected_size=random.randint(50, 60),
                    filler=u"<PAD>")

                cropped_terms = x_feature.ValueVector
                subj_ind = s_ind - x_feature.StartIndex
                obj_ind = t_ind - x_feature.StartIndex

                logger.info(len(terms))
                logger.info(u"Source Index: {}".format(subj_ind))
                logger.info(u"Target Index: {}".format(obj_ind))
                s = u" ".join(terms_to_str(cropped_terms))
                logger.info(u"Result: {}".format(s))

                assert(isinstance(x_feature.ValueVector[subj_ind], Entity))
                assert(isinstance(x_feature.ValueVector[obj_ind], Entity))


if __name__ == '__main__':
    unittest.main()
