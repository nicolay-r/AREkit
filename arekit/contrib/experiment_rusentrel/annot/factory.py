from arekit.common.experiment.annot.default_annotator import DefaultAnnotator
from arekit.contrib.experiment_rusentrel.annot.two_scale import TwoScaleTaskAnnotator


class ExperimentAnnotatorFactory:

    @staticmethod
    def create(labels_count, create_algo):
        assert(isinstance(labels_count, int))
        assert(callable(create_algo))

        if labels_count == 2:
            return TwoScaleTaskAnnotator()
        else:
            return DefaultAnnotator(create_algo())
