#!/usr/bin/python
from core.common.bound import Bound
from core.common.entities.collection import EntityCollection
from core.processing.lemmatization.mystem import MystemWrapper
from core.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.io_utils import RuSentRelIOUtils
from core.source.rusentrel.news import RuSentRelNews
from core.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from core.source.rusentrel.opinions.opinion import RuSentRelOpinion
from core.source.rusentrel.sentence import RuSentRelSentence
from core.source.rusentrel.synonyms import RuSentRelSynonymsCollection

# Initializing stemmer
stemmer = MystemWrapper()

# Reading synonyms collection.
# TODO. Read from zip archive
synonyms = RuSentRelSynonymsCollection.from_file(filepath=RuSentRelIOUtils.get_synonyms_filepath(),
                                                 stemmer=stemmer)

# Reading 'train' subfolder of collection.
train_root = 'test'


for news_id in RuSentRelIOUtils.iter_test_indices():

    print("NewsID: {}".format(news_id))

    # Init filepaths
    entities_filepath = RuSentRelIOUtils.get_entity_filepath(news_id, root=train_root)
    news_filepath = RuSentRelIOUtils.get_news_filepath(news_id, root=train_root)
    opinion_filepath = RuSentRelIOUtils.get_sentiment_opin_filepath(news_id, root=train_root, is_etalon=True)

    # Read collections
    # TODO. Read from zip archive
    entities = RuSentRelDocumentEntityCollection.from_file(entities_filepath, stemmer=stemmer, synonyms=synonyms)
    # TODO. Read from zip archive
    news = RuSentRelNews.from_file(news_filepath, entities)
    # TODO. Read from zip archive
    opininons = RuSentRelOpinionCollection.from_file(opinion_filepath, synonyms=synonyms)

    # Example: Access to the read OPINIONS collection.
    for opinion in opininons:
        assert(isinstance(opinion, RuSentRelOpinion))
        print u"\t{}->{} ({}) [synonym groups opinion: {}->{}]".format(
            opinion.SourceValue,
            opinion.TargetValue,
            opinion.Sentiment.to_str(),
            # Considering synonyms.
            synonyms.get_synonym_group_index(opinion.SourceValue),
            synonyms.get_synonym_group_index(opinion.TargetValue)).encode('utf-8')

    # Example: Access to the read NEWS collection.
    for sentence in news.iter_sentences():
        assert(isinstance(sentence, RuSentRelSentence))
        # Access to text.
        print u"\tSentence: '{}'".format(sentence.Text.strip()).encode('utf-8')
        # Access to inner entities.
        for entity, bound in sentence.iter_entity_with_local_bounds():
            assert(isinstance(entity, RuSentRelEntity))
            assert(isinstance(bound, Bound))
            print u"\tEntity: {}, text position: ({}-{}), IdInDocument: {}".format(
                entity.Value,
                bound.Position,
                bound.Position + bound.Length,
                entity.IdInDocument).encode('utf-8')

    # Example: Access to the read ENTITIES collection.
    example_entity = entities.get_entity_by_index(10)
    entities_list = entities.try_get_entities(example_entity.Value,
                                              group_key=EntityCollection.KeyType.BY_SYNONYMS)
    print u"\tText synonymous to: '{}'".format(example_entity.Value).encode('utf-8')
    print u"\t[{}]".format(", ".join([str((e.Value, str(e.IdInDocument))) for e in entities_list])).encode('utf-8')
