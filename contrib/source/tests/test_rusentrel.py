#!/usr/bin/python
import logging
import sys
import unittest

sys.path.append('../../../../')

from arekit.common.labels.base import Label
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.bound import Bound
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection

from arekit.contrib.source.tests.utils import read_rusentrel_synonyms_collection
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.sentence import RuSentRelSentence
from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


class TestRuSentRel(unittest.TestCase):

    __version = RuSentRelVersions.V11

    def __iter_by_docs(self, synonyms):
        for doc_id in RuSentRelIOUtils.iter_collection_indices(self.__version):

            news = RuSentRelNews.read_document(doc_id=doc_id,
                                               synonyms=synonyms,
                                               version=self.__version)

            opins_it = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)

            opinions = OpinionCollection.init_as_custom(opinions=opins_it,
                                                        synonyms=synonyms)
            yield news, opinions

    def test_reading(self):

        # Initializing logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        synonyms = read_rusentrel_synonyms_collection(version=self.__version)
        for news, opinions in self.__iter_by_docs(synonyms):
            logger.info(u"NewsID: {}".format(news.ID))

            # Example: Access to the read OPINIONS collection.
            for opinion in opinions:
                assert(isinstance(opinion, Opinion))
                logger.info(u"\t{}->{} ({}) [synonym groups opinion: {}->{}]".format(
                    opinion.SourceValue,
                    opinion.TargetValue,
                    opinion.Sentiment.to_class_str(),
                    # Considering synonyms.
                    synonyms.get_synonym_group_index(opinion.SourceValue),
                    synonyms.get_synonym_group_index(opinion.TargetValue)).encode('utf-8'))

            # Example: Access to the read NEWS collection.
            for sentence in news.iter_sentences(return_text=False):
                assert(isinstance(sentence, RuSentRelSentence))
                # Access to text.
                logger.info(u"\tSentence: '{}'".format(sentence.Text.strip()).encode('utf-8'))
                # Access to inner entities.
                for entity, bound in sentence.iter_entity_with_local_bounds():
                    assert(isinstance(entity, RuSentRelEntity))
                    assert(isinstance(bound, Bound))
                    logger.info(u"\tEntity: {} ({}), text position: ({}-{}), IdInDocument: {}".format(
                        entity.Value,
                        entity.Type,
                        bound.Position,
                        bound.Position + bound.Length,
                        entity.IdInDocument).encode('utf-8'))

    def test_linked_text_opinion_extraction(self):

        # Initializing logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        synonyms = read_rusentrel_synonyms_collection(self.__version)
        for news, opinions in self.__iter_by_docs(synonyms):

            logger.info(u"NewsID: {}".format(news.ID))

            # Example: Access to news text-level opinions.
            first_opinion = opinions[0]
            assert(isinstance(first_opinion, Opinion))

            print u"'{src}'->'{tgt}'".format(src=first_opinion.SourceValue,
                                             tgt=first_opinion.TargetValue).encode('utf-8')

            linked_text_opinions = news.extract_linked_text_opinions(first_opinion)
            assert(isinstance(linked_text_opinions, LinkedTextOpinionsWrapper))
            print "Linked opinions count: {}".format(len(linked_text_opinions))
            for text_opinion in linked_text_opinions:
                assert(isinstance(text_opinion, TextOpinion))
                label = text_opinion.Sentiment
                assert(isinstance(label, Label))
                print "<{},{},{}>".format(text_opinion.SourceId, text_opinion.TargetId, str(label))


if __name__ == '__main__':
    unittest.main()
