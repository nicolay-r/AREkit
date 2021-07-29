from arekit.common.labels.base import Label


def label_to_str(label):
    assert(isinstance(label, Label))
    return label.to_class_str()
