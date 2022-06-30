from arekit.common.labels.base import Label


def label_to_str(label):
    assert(isinstance(label, Label))
    return label.to_class_str()


def check_is_supported(label, is_label_supported):
    if label is None:
        return True

    if not is_label_supported(label):
        raise Exception("Label \"{label}\" is not supported in evaluator!".format(label=label_to_str(label)))
