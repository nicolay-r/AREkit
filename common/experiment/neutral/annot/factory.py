from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator


def create_annotator(labels_count, dist_in_terms_between_opin_ends=None):
    assert(isinstance(labels_count, int))
    assert(isinstance(dist_in_terms_between_opin_ends, int) or dist_in_terms_between_opin_ends is None)

    annot_type = get_annotator_type(labels_count)

    if labels_count == 2:
        return annot_type()
    elif labels_count == 3:
        return annot_type(dist_in_terms_between_opin_ends)
    raise NotImplementedError(u"Could not create neutral annotator instance for scaler '{}'".format(str(labels_count)))


def get_annotator_type(labels_count):
    if labels_count == 2:
        return TwoScaleNeutralAnnotator
    elif labels_count == 3:
        return ThreeScaleNeutralAnnotator
    raise NotImplementedError(u"Could not create neutral annotator type for scaler '{}'".format(str(labels_count)))
