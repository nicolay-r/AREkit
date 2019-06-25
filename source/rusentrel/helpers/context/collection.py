from core.common.opinions.collection import OpinionCollection
from core.common.opinions.opinion import Opinion
from core.common.text_opinions.collection import TextOpinionCollection
from core.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.helpers.context.opinion import RuSentRelTextOpinion


class RuSentRelTextOpinionCollection(TextOpinionCollection):

    def __init__(self, text_opinions):
        super(RuSentRelTextOpinionCollection, self).__init__(
            parsed_news_collection=None,
            text_opinions=text_opinions)

    @classmethod
    def from_opinions(cls,
                      rusentrel_news_id,
                      doc_entities,
                      opinions,
                      check_text_opinion_correctness,
                      debug=False):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(opinions, OpinionCollection))
        text_opinions = []
        for opinion in opinions:
            text_opinions.extend(
                cls.__from_opinion(
                    rusentrel_news_id=rusentrel_news_id,
                    doc_entities=doc_entities,
                    opinion=opinion,
                    check_text_opinion_correctness=check_text_opinion_correctness,
                    debug=debug))

        return cls(text_opinions)

    @classmethod
    def from_opinion(cls,
                     rusentrel_news_id,
                     doc_entities,
                     opinion,
                     check_text_opinion_correctness,
                     debug=False):
        return cls(cls.__from_opinion(
            rusentrel_news_id=rusentrel_news_id,
            doc_entities=doc_entities,
            opinion=opinion,
            check_text_opinion_correctness=check_text_opinion_correctness,
            debug=debug))

    @staticmethod
    def __from_opinion(
            rusentrel_news_id,
            doc_entities,
            opinion,
            check_text_opinion_correctness,
            debug=False):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(doc_entities, RuSentRelDocumentEntityCollection))
        assert(isinstance(opinion, Opinion))
        assert(callable(check_text_opinion_correctness) or
               check_text_opinion_correctness is None)

        source_entities = doc_entities.try_get_entities(
            opinion.SourceValue, group_key=RuSentRelDocumentEntityCollection.KeyType.BY_SYNONYMS)
        target_entities = doc_entities.try_get_entities(
            opinion.TargetValue, group_key=RuSentRelDocumentEntityCollection.KeyType.BY_SYNONYMS)

        if source_entities is None:
            if debug:
                print "Appropriate entity for '{}'->'...' has not been found".format(
                    opinion.SourceValue.encode('utf-8'))
            return []

        if target_entities is None:
            if debug:
                print "Appropriate entity for '...'->'{}' has not been found".format(
                    opinion.TargetValue.encode('utf-8'))
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
                    doc_entities=doc_entities)

                if check_text_opinion_correctness is not None:
                    if not check_text_opinion_correctness(text_opinion):
                        continue

                text_opinions.append(text_opinion)

        return text_opinions
