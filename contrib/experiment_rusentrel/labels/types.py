from arekit.common.labels.base import Label, NoLabel


class ExperimentNeutralLabel(NoLabel):
    pass


class ExperimentNegativeLabel(Label):
    """ RuSentRel Experiment Positive Label.
    """
    pass


class ExperimentPositiveLabel(Label):
    """ RuSentRel Experiment Negative Label.
    """
    pass
