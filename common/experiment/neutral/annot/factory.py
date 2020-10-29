from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator


def create_annotator(labels_count, dist_in_terms_between_opin_ends=None):
    assert(isinstance(labels_count, int))
    assert(isinstance(dist_in_terms_between_opin_ends, int) or dist_in_terms_between_opin_ends is None)

    if labels_count == 2:
        return TwoScaleNeutralAnnotator()
    elif labels_count == 3:
        return ThreeScaleNeutralAnnotator(dist_in_terms_between_opin_ends)
    raise NotImplementedError(u"Could not create neutral annotator for scaler '{}'".format(str(labels_count)))

