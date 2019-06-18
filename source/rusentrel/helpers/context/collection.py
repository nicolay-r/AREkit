from core.common.opinions.opinion import Opinion
from core.source.rusentrel.entities.collection import RuSentRelEntityCollection
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.news import RuSentRelNews
from core.source.rusentrel.helpers.context.opinion import RuSentRelContextOpinion


class RuSentRelContextOpinionList:

    def __init__(self, opinion_list):
        assert(isinstance(opinion_list, list))
        self.__ctx_opinions = opinion_list

    @classmethod
    def from_news_opinion(cls, news, opinion, debug=False):
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinion, Opinion))

        doc_entities = news.DocEntities
        assert(isinstance(doc_entities, RuSentRelEntityCollection))

        source_entities = doc_entities.try_get_entities(
            opinion.SourceValue, group_key=RuSentRelEntityCollection.KeyType.BY_SYNONYMS)
        target_entities = doc_entities.try_get_entities(
            opinion.TargetValue, group_key=RuSentRelEntityCollection.KeyType.BY_SYNONYMS)

        if source_entities is None:
            if debug:
                print "Appropriate entity for '{}'->'...' has not been found".format(
                    opinion.SourceValue.encode('utf-8'))
            return cls(opinion_list=[])

        if target_entities is None:
            if debug:
                print "Appropriate entity for '...'->'{}' has not been found".format(
                    opinion.TargetValue.encode('utf-8'))
            return cls(opinion_list=[])

        ctx_opinions = []
        for source_entity in source_entities:
            for target_entity in target_entities:
                assert(isinstance(source_entity, RuSentRelEntity))
                assert(isinstance(target_entity, RuSentRelEntity))
                relation = RuSentRelContextOpinion(e_source_doc_level_id=source_entity.IdInDocument,
                                                   e_target_doc_level_id=target_entity.IdInDocument,
                                                   doc_entities=doc_entities)
                ctx_opinions.append(relation)

        return cls(ctx_opinions)

    def apply_filter(self, filter_function):
        self.__ctx_opinions = [r for r in self.__ctx_opinions if filter_function(r)]

    def __getitem__(self, item):
        assert(isinstance(item,  int))
        return self.__ctx_opinions[item]

    def __len__(self):
        return len(self.__ctx_opinions)

    def __iter__(self):
        for opinion in self.__ctx_opinions:
            yield opinion