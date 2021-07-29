from arekit.contrib.experiment_rusentrel.annot.three_scale import ThreeScaleTaskAnnotator
from arekit.contrib.experiment_rusentrel.annot.two_scale import TwoScaleTaskAnnotator


class ExperimentAnnotatorFactory:

    @staticmethod
    def create(labels_count, create_algo):
        assert(isinstance(labels_count, int))
        assert(callable(create_algo))

        if labels_count == 2:
            return TwoScaleTaskAnnotator()
        elif labels_count == 3:
            return ThreeScaleTaskAnnotator(create_algo())
        else:
            raise NotImplementedError()
