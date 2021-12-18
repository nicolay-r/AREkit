from arekit.common.entities.base import Entity
from arekit.common.experiment.annot.base_annot import BaseAnnotationAlgorithm
from arekit.common.labels.base import Label
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.opinion_pairs import OpinionPairsProvider
from arekit.common.opinions.base import Opinion
from arekit.common.dataset.text_opinions.enums import DistanceType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper


class DefaultSingleLabelAnnotationAlgorithm(BaseAnnotationAlgorithm):
    """
    Neutral annotation algorithm which assumes to compose pairs
    within a sentence which are not a part of sentiment.
    """

    def __init__(self, dist_in_terms_bound, label_instance, dist_in_sents=0, ignored_entity_values=None):
        """
        dist_in_terms_bound: int
            max allowed distance in term (less than passed value)
        """
        assert(isinstance(ignored_entity_values, list) or ignored_entity_values is None)
        assert(isinstance(dist_in_terms_bound, int) or dist_in_terms_bound is None)
        assert(isinstance(label_instance, Label))
        assert(isinstance(dist_in_sents, int))

        self.__ignored_entity_values = [] if ignored_entity_values is None else ignored_entity_values
        self.__dist_in_terms_bound = dist_in_terms_bound
        self.__dist_in_sents = dist_in_sents
        self.__label_instance = label_instance

    # region private methods

    @staticmethod
    def __create_key_by_entity_pair(e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))
        return "{}_{}".format(e1.IdInDocument, e2.IdInDocument)

    def __is_ignored_entity_value(self, entity_value):
        assert(isinstance(entity_value, str))
        return entity_value in self.__ignored_entity_values

    def __try_create_pair_key(self, parsed_news, e1, e2, existed_opinions):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        if e1.IdInDocument == e2.IdInDocument:
            return

        if self.__is_ignored_entity_value(entity_value=e1.Value):
            return
        if self.__is_ignored_entity_value(entity_value=e2.Value):
            return

        s_dist = TextOpinionHelper.calc_dist_between_entities(parsed_news=parsed_news,
                                                              e1=e1, e2=e2,
                                                              distance_type=DistanceType.InSentences)

        if s_dist > self.__dist_in_sents:
            return

        t_dist = TextOpinionHelper.calc_dist_between_entities(parsed_news=parsed_news,
                                                              e1=e1, e2=e2,
                                                              distance_type=DistanceType.InTerms)

        if self.__dist_in_terms_bound is not None and t_dist > self.__dist_in_terms_bound:
            return

        if existed_opinions is not None:
            o = Opinion(source_value=e1.Value,
                        target_value=e2.Value,
                        sentiment=self.__label_instance)
            if existed_opinions.has_synonymous_opinion(opinion=o):
                return

        return self.__create_key_by_entity_pair(e1=e1, e2=e2)

    # endregion

    def iter_opinions(self, parsed_news, existed_opinions=None):
        assert(isinstance(parsed_news, ParsedNews))

        def __filter_pair_func(e1, e2):
            key = self.__try_create_pair_key(
                parsed_news=parsed_news,
                e1=e1, e2=e2,
                existed_opinions=existed_opinions)

            return key is not None

        # Init opinion provider.
        opinions_provider = OpinionPairsProvider(parsed_news=parsed_news)

        return opinions_provider.iter_from_all(
            label=self.__label_instance,
            filter_func=__filter_pair_func)
