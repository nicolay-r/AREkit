from arekit.common.labels.provider.base import BasePairLabelProvider
from arekit.common.news.entity import DocumentEntity
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider, DistanceType
from arekit.common.news.parsed.providers.opinion_pairs import OpinionPairsProvider
from arekit.common.opinions.annot.algo.base import BaseAnnotationAlgorithm
from arekit.common.opinions.base import Opinion


class PairBasedAnnotationAlgorithm(BaseAnnotationAlgorithm):
    """ Is a pair-based annotation algorithm which
        assumes to compose source-target entity pairs
    """

    def __init__(self, dist_in_terms_bound, label_provider, dist_in_sents=0, ignored_entity_values=None):
        """
        dist_in_terms_bound: int
            max allowed distance in term (less than passed value)
        """
        assert(isinstance(dist_in_terms_bound, int) or dist_in_terms_bound is None)
        assert(isinstance(label_provider, BasePairLabelProvider))
        assert(isinstance(dist_in_sents, int))
        assert(isinstance(ignored_entity_values, list) or ignored_entity_values is None)

        self.__ignored_entity_values = [] if ignored_entity_values is None else ignored_entity_values
        self.__label_provider = label_provider
        self.__dist_in_terms_bound = dist_in_terms_bound
        self.__dist_in_sents = dist_in_sents

    # region private methods

    @staticmethod
    def __create_key_by_entity_pair(e1, e2):
        assert(isinstance(e1, DocumentEntity))
        assert(isinstance(e2, DocumentEntity))
        return "{}_{}".format(e1.IdInDocument, e2.IdInDocument)

    def __is_ignored_entity_value(self, entity_value):
        assert(isinstance(entity_value, str))
        return entity_value in self.__ignored_entity_values

    def __try_create_pair_key(self, entity_service, e1, e2, existed_opinions):
        assert(isinstance(entity_service, EntityServiceProvider))
        assert(isinstance(e1, DocumentEntity))
        assert(isinstance(e2, DocumentEntity))

        if e1.IdInDocument == e2.IdInDocument:
            return

        if self.__is_ignored_entity_value(entity_value=e1.Value):
            return
        if self.__is_ignored_entity_value(entity_value=e2.Value):
            return

        s_dist = entity_service.calc_dist_between_entities(e1=e1, e2=e2, distance_type=DistanceType.InSentences)

        if s_dist > self.__dist_in_sents:
            return

        t_dist = entity_service.calc_dist_between_entities(e1=e1, e2=e2, distance_type=DistanceType.InTerms)

        if self.__dist_in_terms_bound is not None and t_dist > self.__dist_in_terms_bound:
            return

        if existed_opinions is not None:
            o = Opinion(source_value=e1.Value,
                        target_value=e2.Value,
                        sentiment=self.__label_provider.provide(source=e1, target=e2))
            if existed_opinions.has_synonymous_opinion(opinion=o):
                return

        return self.__create_key_by_entity_pair(e1=e1, e2=e2)

    # endregion

    def iter_opinions(self, parsed_news, existed_opinions=None):
        assert(isinstance(parsed_news, ParsedNews))

        def __filter_pair_func(e1, e2):
            key = self.__try_create_pair_key(entity_service=entity_service_provider,
                                             e1=e1, e2=e2,
                                             existed_opinions=existed_opinions)

            return key is not None

        # Initialize providers.
        # TODO. Provide here service #245 issue.
        opinions_provider = OpinionPairsProvider()
        entity_service_provider = EntityServiceProvider()
        opinions_provider.init_parsed_news(parsed_news)
        entity_service_provider.init_parsed_news(parsed_news)

        return opinions_provider.iter_from_all(label_provider=self.__label_provider,
                                               filter_func=__filter_pair_func)
