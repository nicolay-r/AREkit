from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir


def get_neutral_annotation_root(experiment):
    assert(isinstance(experiment, BaseExperiment))
    return get_path_of_subfolder_in_experiments_dir(
        experiments_dir=experiment.DataIO.get_input_samples_dir(experiment.Name),
        subfolder_name=__get_annot_name(experiment.DataIO.NeutralAnnotator))


def __get_annot_name(neutral_annot):
    assert(isinstance(neutral_annot, BaseNeutralAnnotator))
    if isinstance(neutral_annot, TwoScaleNeutralAnnotator):
        return u"neut_2_scale"
    if isinstance(neutral_annot, ThreeScaleNeutralAnnotator):
        return u"neut_3_scale"
