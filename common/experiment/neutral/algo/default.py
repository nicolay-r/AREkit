from arekit.common.entities.base import Entity
from arekit.common.entities.collection import EntityCollection
from arekit.common.experiment.neutral.algo.base import BaseNeutralAnnotationAlgorithm
from arekit.common.labels.base import NeutralLabel
from arekit.common.news.parsed.collection import ParsedNewsCollection
from arekit.common.opinions.base import Opinion
from arekit.common.dataset.text_opinions.enums import DistanceType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper


class DefaultNeutralAnnotationAlgorithm(BaseNeutralAnnotationAlgorithm):
    """
    Neutral annotation algorithm which assumes to compose pairs
    within a sentence which are not a part of sentiment.
    """

    def __init__(self,
                 iter_parsed_news,
                 dist_in_terms_bound,
                 ignored_entity_values=None):
        """
        dist_in_terms_bound: int
            max allowed distance in term (less than passed value)
        """
        assert(isinstance(ignored_entity_values, list) or ignored_entity_values is None)
        assert(isinstance(dist_in_terms_bound, int))

        self.__ignored_entity_values = [] if ignored_entity_values is None else ignored_entity_values

        self.__pnc = ParsedNewsCollection(parsed_news_it=iter_parsed_news, notify=True)
        self.__text_opinion_helper = TextOpinionHelper(lambda news_id: self.__pnc.get_by_news_id(news_id))
        self.__dist_in_terms_bound = dist_in_terms_bound

    # region private methods

    @staticmethod
    def __create_key_by_entity_pair(e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))
        return u"{}_{}".format(e1.IdInDocument, e2.IdInDocument)

    def __is_ignored_entity_value(self, entity_value):
        assert(isinstance(entity_value, unicode))
        return entity_value in self.__ignored_entity_values

    def __iter_opinions_between_entties(self, relevant_pairs, entities_collection):
        assert(isinstance(entities_collection, EntityCollection))

        for e1 in entities_collection:
            assert(isinstance(e1, Entity))

            for e2 in entities_collection:
                assert(isinstance(e2, Entity))

                key = self.__create_key_by_entity_pair(e1=e1, e2=e2)
                if key not in relevant_pairs:
                    continue

                opinion = Opinion(source_value=e1.Value,
                                  target_value=e2.Value,
                                  sentiment=NeutralLabel())

                yield opinion

    def __try_create_pair_key(self, news_id, e1, e2, sentiment_opinions):
        assert(isinstance(news_id, int))
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        if e1.IdInDocument == e2.IdInDocument:
            return

        if self.__is_ignored_entity_value(entity_value=e1.Value):
            return
        if self.__is_ignored_entity_value(entity_value=e2.Value):
            return

        s_dist = self.__text_opinion_helper.calc_dist_between_entities(
            news_id=news_id, e1=e1, e2=e2, distance_type=DistanceType.InSentences)

        if s_dist > 0:
            return

        t_dist = self.__text_opinion_helper.calc_dist_between_entities(
            news_id=news_id, e1=e1, e2=e2, distance_type=DistanceType.InTerms)

        if t_dist > self.__dist_in_terms_bound:
            return

        if sentiment_opinions is not None:
            o = Opinion(source_value=e1.Value,
                        target_value=e2.Value,
                        sentiment=NeutralLabel())
            if sentiment_opinions.has_synonymous_opinion(opinion=o):
                return

        return self.__create_key_by_entity_pair(e1=e1, e2=e2)

    # endregion

    def iter_neutral_opinions(self, news_id, entities_collection, sentiment_opinions=None):
        assert(isinstance(news_id, int))
        assert(isinstance(entities_collection, EntityCollection))

        relevant_pairs = {}

        for e1 in entities_collection:
            assert(isinstance(e1, Entity))

            for e2 in entities_collection:
                assert(isinstance(e2, Entity))

                key = self.__try_create_pair_key(news_id=news_id, e1=e1, e2=e2, sentiment_opinions=sentiment_opinions)

                if key is None:
                    continue

                relevant_pairs[key] = 0

        return self.__iter_opinions_between_entties(relevant_pairs=relevant_pairs,
                                                    entities_collection=entities_collection)
