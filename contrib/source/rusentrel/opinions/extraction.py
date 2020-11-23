import collections
import logging

from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity

logger = logging.getLogger(__name__)


# region private methods

def __from_opinion(rusentrel_news_id, source_entities, target_entities, opinion):

    for source_entity in source_entities:
        for target_entity in target_entities:
            assert (isinstance(source_entity, RuSentRelEntity))
            assert (isinstance(target_entity, RuSentRelEntity))

            text_opinion = TextOpinion(news_id=rusentrel_news_id,
                                       source_id=source_entity.IdInDocument,
                                       target_id=target_entity.IdInDocument,
                                       label=opinion.Sentiment,
                                       owner=None,
                                       text_opinion_id=None)

            yield text_opinion

# endregion


def iter_text_opinions_by_doc_opinion(rusentrel_news_id, doc_entities, opinion, debug=False):
    """ Provides text-level opinion extraction by document-level opinions
        (Opinion class instances), for a particular document (news_id),
        with the realated entity collection.
    """
    assert(isinstance(rusentrel_news_id, int))
    assert(isinstance(opinion, Opinion))
    assert(isinstance(doc_entities, RuSentRelDocumentEntityCollection))

    source_entities = doc_entities.try_get_entities(
        opinion.SourceValue, group_key=RuSentRelDocumentEntityCollection.KeyType.BY_SYNONYMS)
    target_entities = doc_entities.try_get_entities(
        opinion.TargetValue, group_key=RuSentRelDocumentEntityCollection.KeyType.BY_SYNONYMS)

    if source_entities is None:
        if debug:
            logger.info("Appropriate entity for '{}'->'...' has not been found".format(
                opinion.SourceValue.encode('utf-8')))
        return
        yield

    if target_entities is None:
        if debug:
            logger.info("Appropriate entity for '...'->'{}' has not been found".format(
                opinion.TargetValue.encode('utf-8')))
        return
        yield

    text_opins_iter = __from_opinion(rusentrel_news_id=rusentrel_news_id,
                                     source_entities=source_entities,
                                     target_entities=target_entities,
                                     opinion=opinion)

    for text_opinion in text_opins_iter:
        yield text_opinion
