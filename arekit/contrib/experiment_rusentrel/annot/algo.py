from arekit.common.entities.base import Entity
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel


class RuSentRelDefaultNeutralAnnotationAlgorithm(PairBasedAnnotationAlgorithm):

    IGNORED_ENTITY_VALUES = ["author", "unknown"]

    @staticmethod
    def __is_entity_ignored(e, e_type):
        assert(isinstance(e, Entity))
        return e.Value in RuSentRelDefaultNeutralAnnotationAlgorithm.IGNORED_ENTITY_VALUES

    def __init__(self, dist_in_terms_bound):
        label_provider = ConstantLabelProvider(label_instance=ExperimentNeutralLabel())
        super(RuSentRelDefaultNeutralAnnotationAlgorithm, self).__init__(
            dist_in_sents=0,
            dist_in_terms_bound=dist_in_terms_bound,
            label_provider=label_provider,
            is_entity_ignored_func=self.__is_entity_ignored)
