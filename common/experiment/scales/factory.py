from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler


def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 2:
        return TwoLabelScaler()
    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")
