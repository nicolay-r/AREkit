#!/usr/bin/python
import logging
from arekit.common.bound import Bound
from arekit.common.opinions.base import Opinion
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news.base import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.sentence import RuSentRelSentence
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection


# Initializing logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)

# Initializing stemmer
stemmer = MystemWrapper()

# Reading synonyms collection.
synonyms = RuSentRelSynonymsCollection.load_collection(stemmer=stemmer)

for doc_id in RuSentRelIOUtils.iter_collection_indices():

    logger.info(u"NewsID: {}".format(doc_id))

    news = RuSentRelNews.read_document(doc_id=doc_id,
                                       synonyms=synonyms)

    opinions = RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                          synonyms=synonyms)

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
    for sentence in news.iter_sentences():
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
