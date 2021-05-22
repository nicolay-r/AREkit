from arekit.contrib.experiment_rusentrel.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.contrib.experiment_rusentrel.annot.two_scale import TwoScaleNeutralAnnotator


class ExperimentNeutralAnnotatorFactory:

    @staticmethod
    def create(labels_count, create_algo):
        assert(isinstance(labels_count, int))
        assert(callable(create_algo))

        if labels_count == 2:
            return TwoScaleNeutralAnnotator()
        elif labels_count == 3:
            return ThreeScaleNeutralAnnotator(create_algo())
        else:
            raise NotImplementedError()
