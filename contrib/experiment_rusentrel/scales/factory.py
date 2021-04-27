from arekit.contrib.experiment_rusentrel.scales.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.scales.two import TwoLabelScaler


# TODO. Move this factory out of this experiment.
# TODO. Into project.
def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 2:
        return TwoLabelScaler()
    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")
