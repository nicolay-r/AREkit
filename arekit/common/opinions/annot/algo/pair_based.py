from arekit.common.entities.types import OpinionEntityType
from arekit.common.labels.provider.base import BasePairLabelProvider
from arekit.common.docs.entity import DocumentEntity
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider, DistanceType
from arekit.common.docs.parsed.providers.opinion_pairs import OpinionPairsProvider
from arekit.common.opinions.annot.algo.base import BaseOpinionAnnotationAlgorithm
from arekit.common.opinions.base import Opinion


class PairBasedOpinionAnnotationAlgorithm(BaseOpinionAnnotationAlgorithm):
    """ Is a pair-based annotation algorithm which assumes to compose source-target entity pairs
        This is a default annotator which found its application in Sentiment Attitude Extraction task [1].

        References:
            [1] Extracting Sentiment Attitudes from Analytical Texts https://arxiv.org/pdf/1808.08932.pdf
    """

    def __init__(self, dist_in_terms_bound, label_provider, entity_index_func, dist_in_sents=0,
                 is_entity_ignored_func=None):
        """
        dist_in_terms_bound: int
            max allowed distance in term (less than passed value)
        is_entity_ignored_func: func
            entity, type -> bool
        """
        assert(isinstance(dist_in_terms_bound, int) or dist_in_terms_bound is None)
        assert(isinstance(label_provider, BasePairLabelProvider))
        assert(callable(entity_index_func))
        assert(isinstance(dist_in_sents, int))
        assert(callable(is_entity_ignored_func) or is_entity_ignored_func is None)

        self.__label_provider = label_provider
        self.__dist_in_terms_bound = dist_in_terms_bound
        self.__dist_in_sents = dist_in_sents
        self.__is_entity_ignored_func = is_entity_ignored_func
        self.__entity_index_func = entity_index_func

    # region private methods

    @staticmethod
    def __create_key_by_entity_pair(e1, e2):
        assert(isinstance(e1, DocumentEntity))
        assert(isinstance(e2, DocumentEntity))
        return "{}_{}".format(e1.IdInDocument, e2.IdInDocument)

    def __try_create_pair_key(self, entity_service, e1, e2, existed_opinions):
        assert(isinstance(entity_service, EntityServiceProvider))
        assert(isinstance(e1, DocumentEntity))
        assert(isinstance(e2, DocumentEntity))

        if e1.IdInDocument == e2.IdInDocument:
            return

        if self.__is_entity_ignored_func is not None:
            if self.__is_entity_ignored_func(e1, OpinionEntityType.Subject):
                return
            if self.__is_entity_ignored_func(e2, OpinionEntityType.Object):
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
                        label=self.__label_provider.provide(source=e1, target=e2))
            if existed_opinions.has_synonymous_opinion(opinion=o):
                return

        return self.__create_key_by_entity_pair(e1=e1, e2=e2)

    # endregion

    def iter_opinions(self, parsed_doc, existed_opinions=None):
        assert(isinstance(parsed_doc, ParsedDocument))

        def __filter_pair_func(e1, e2):
            key = self.__try_create_pair_key(entity_service=entity_service_provider,
                                             e1=e1, e2=e2,
                                             existed_opinions=existed_opinions)

            return key is not None

        # Initialize providers.
        opinions_provider = OpinionPairsProvider(entity_index_func=self.__entity_index_func)
        entity_service_provider = EntityServiceProvider(entity_index_func=self.__entity_index_func)
        opinions_provider.init_parsed_doc(parsed_doc)
        entity_service_provider.init_parsed_doc(parsed_doc)

        return opinions_provider.iter_from_all(label_provider=self.__label_provider,
                                               filter_func=__filter_pair_func)
