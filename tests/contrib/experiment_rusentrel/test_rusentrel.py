#!/usr/bin/python
import logging
import sys
import unittest


sys.path.append('../../../../')

from arekit.common.labels.base import Label
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.bound import Bound
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parser import NewsParser
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.text.parser import BaseTextParser

from arekit.processing.lemmatization.mystem import MystemWrapper

from arekit.contrib.source.rusentrel.news_reader import RuSentRelNews
from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.sentence import RuSentRelSentence
from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


class TestRuSentRel(unittest.TestCase):

    __version = RuSentRelVersions.V11

    @staticmethod
    def __read_rusentrel_synonyms_collection():
        # Initializing stemmer
        stemmer = MystemWrapper()

        # Reading synonyms collection.
        return RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer,
                                                                   version=TestRuSentRel.__version)

    def __iter_by_docs(self, synonyms):
        for doc_id in RuSentRelIOUtils.iter_collection_indices(self.__version):

            news = RuSentRelNews.read_document(doc_id=doc_id,
                                               synonyms=synonyms,
                                               version=self.__version)

            opins_it = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)

            opinions = OpinionCollection(opinions=opins_it,
                                         synonyms=synonyms,
                                         error_on_duplicates=True,
                                         error_on_synonym_end_missed=True)
            yield news, opinions

    def test_reading(self):

        # Initializing logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        synonyms = TestRuSentRel.__read_rusentrel_synonyms_collection()
        for news, opinions in self.__iter_by_docs(synonyms):
            logger.info("NewsID: {}".format(news.ID))

            # Example: Access to the read OPINIONS collection.
            for opinion in opinions:
                assert(isinstance(opinion, Opinion))
                logger.info("\t{}->{} ({}) [synonym groups opinion: {}->{}]".format(
                    opinion.SourceValue,
                    opinion.TargetValue,
                    opinion.Sentiment.to_class_str(),
                    # Considering synonyms.
                    synonyms.get_synonym_group_index(opinion.SourceValue),
                    synonyms.get_synonym_group_index(opinion.TargetValue)))

            # Example: Access to the read NEWS collection.
            for sentence in news.iter_sentences():
                assert(isinstance(sentence, RuSentRelSentence))
                # Access to text.
                logger.info("\tSentence: '{}'".format(sentence.Text.strip()))
                # Access to inner entities.
                for entity, bound in sentence.iter_entity_with_local_bounds():
                    assert(isinstance(entity, RuSentRelEntity))
                    assert(isinstance(bound, Bound))
                    logger.info("\tEntity: {} ({}), text position: ({}-{}), ID: {}".format(
                        entity.Value,
                        entity.Type,
                        bound.Position,
                        bound.Position + bound.Length,
                        entity.ID))

    def test_linked_text_opinion_extraction(self):

        # Initializing logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        # Init text parser.
        text_parser = BaseTextParser(pipeline=[RuSentRelTextEntitiesParser()])

        synonyms = TestRuSentRel.__read_rusentrel_synonyms_collection()
        for news, opinions in self.__iter_by_docs(synonyms):

            logger.info("NewsID: {}".format(news.ID))

            # Example: Access to news text-level opinions.
            first_opinion = opinions[0]
            assert(isinstance(first_opinion, Opinion))

            print("'{src}'->'{tgt}'".format(src=first_opinion.SourceValue,
                                            tgt=first_opinion.TargetValue))

            # Parse text.
            parsed_news = NewsParser.parse(news=news, text_parser=text_parser)

            # Initialize text opinion provider.
            text_opinion_provider = TextOpinionPairsProvider(synonyms.get_synonym_group_index)
            text_opinion_provider.init_parsed_news(parsed_news)

            text_opins_it = text_opinion_provider.iter_from_opinion(opinion=first_opinion)

            # Obtain text opinions linkage.
            text_opinons_linkage = TextOpinionsLinkage(text_opins_it)

            print("Linked opinions count: {}".format(len(text_opinons_linkage)))
            for text_opinion in text_opinons_linkage:
                assert(isinstance(text_opinion, TextOpinion))
                label = text_opinion.Sentiment
                assert(isinstance(label, Label))
                print("<{},{},{}>".format(text_opinion.SourceId, text_opinion.TargetId, str(label)))


if __name__ == '__main__':
    unittest.main()
