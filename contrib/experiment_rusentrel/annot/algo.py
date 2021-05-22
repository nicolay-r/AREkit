from arekit.common.experiment.neutral.algo.default import DefaultNeutralAnnotationAlgorithm


class RuSentRelDefaultNeutralAnnotationAlgorithm(DefaultNeutralAnnotationAlgorithm):

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, dist_in_terms_bound):
        super(RuSentRelDefaultNeutralAnnotationAlgorithm, self).__init__(
            dist_in_sents=0,
            dist_in_terms_bound=dist_in_terms_bound,
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)