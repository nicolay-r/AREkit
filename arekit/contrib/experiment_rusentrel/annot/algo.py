from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel


class RuSentRelDefaultNeutralAnnotationAlgorithm(DefaultSingleLabelAnnotationAlgorithm):

    IGNORED_ENTITY_VALUES = ["author", "unknown"]

    def __init__(self, dist_in_terms_bound):
        super(RuSentRelDefaultNeutralAnnotationAlgorithm, self).__init__(
            dist_in_sents=0,
            dist_in_terms_bound=dist_in_terms_bound,
            label_instance=ExperimentNeutralLabel(),
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)