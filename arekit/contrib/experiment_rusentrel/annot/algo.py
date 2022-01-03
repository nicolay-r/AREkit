from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.labels.provider.single_label import PairSingleLabelProvider
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel


class RuSentRelDefaultNeutralAnnotationAlgorithm(PairBasedAnnotationAlgorithm):

    IGNORED_ENTITY_VALUES = ["author", "unknown"]

    def __init__(self, dist_in_terms_bound):
        label_provider = PairSingleLabelProvider(label_instance=ExperimentNeutralLabel())
        super(RuSentRelDefaultNeutralAnnotationAlgorithm, self).__init__(
            dist_in_sents=0,
            dist_in_terms_bound=dist_in_terms_bound,
            label_provider=label_provider,
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)
