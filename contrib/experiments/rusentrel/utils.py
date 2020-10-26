from arekit.contrib.experiments.rusentrel.folding_type import FoldingType


def folding_type_to_str(folding_type):
    assert (isinstance(folding_type, FoldingType))
    if folding_type == FoldingType.Fixed:
        return u"fixed"
    if folding_type == FoldingType.CrossValidation:
        return u"cv"

