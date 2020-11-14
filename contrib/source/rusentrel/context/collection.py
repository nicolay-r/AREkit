import collections
import logging
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.collection import TextOpinionCollection
from arekit.contrib.source.rusentrel.context.opinion import RuSentRelTextOpinion
from arekit.contrib.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity


logger = logging.getLogger(__name__)


class RuSentRelTextOpinionCollection(TextOpinionCollection):
    """
    Collection of a text-level opinions for RuSentRel source.
    """

    # region constructors

    def __init__(self, text_opinions):
        super(RuSentRelTextOpinionCollection, self).__init__(text_opinions=text_opinions)

    @classmethod
    def from_opinions(cls,
                      rusentrel_news_id,
                      doc_entities,
                      opinions,
                      debug=False):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(opinions, collections.Iterable))
        text_opinions = []
        for opinion in opinions:
            text_opinions.extend(
                cls.__from_opinion(rusentrel_news_id=rusentrel_news_id,
                                   doc_entities=doc_entities,
                                   opinion=opinion,
                                   debug=debug))

        return cls(text_opinions)

    # endregion

    # region private methods

    @staticmethod
    def __from_opinion(
            rusentrel_news_id,
            doc_entities,
            opinion,
            debug=False):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(doc_entities, RuSentRelDocumentEntityCollection))
        assert(isinstance(opinion, Opinion))

        source_entities = doc_entities.try_get_entities(
            opinion.SourceValue, group_key=RuSentRelDocumentEntityCollection.KeyType.BY_SYNONYMS)
        target_entities = doc_entities.try_get_entities(
            opinion.TargetValue, group_key=RuSentRelDocumentEntityCollection.KeyType.BY_SYNONYMS)

        if source_entities is None:
            if debug:
                logger.info("Appropriate entity for '{}'->'...' has not been found".format(
                    opinion.SourceValue.encode('utf-8')))
            return []

        if target_entities is None:
            if debug:
                logger.info("Appropriate entity for '...'->'{}' has not been found".format(
                    opinion.TargetValue.encode('utf-8')))
            return []

        text_opinions = []
        for source_entity in source_entities:
            for target_entity in target_entities:
                assert(isinstance(source_entity, RuSentRelEntity))
                assert(isinstance(target_entity, RuSentRelEntity))

                text_opinion = RuSentRelTextOpinion(
                    rusentrel_news_id=rusentrel_news_id,
                    e_source_doc_level_id=source_entity.IdInDocument,
                    e_target_doc_level_id=target_entity.IdInDocument,
                    sentiment=opinion.Sentiment)

                text_opinions.append(text_opinion)

        return text_opinions

    #endregion
