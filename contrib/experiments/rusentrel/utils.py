from arekit.contrib.experiments.rusentrel.folding_type import FoldingType
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils


def folding_type_to_str(folding_type):
    assert (isinstance(folding_type, FoldingType))
    if folding_type == FoldingType.Fixed:
        return u"fixed"
    if folding_type == FoldingType.CrossValidation:
        return u"cv"


def get_rusentrel_inds(version):
    """ Provides all news_inds utilized in RuSentRel collection
    """
    train_doc_ids = list(RuSentRelIOUtils.iter_train_indices(version))
    test_doc_ids = list(RuSentRelIOUtils.iter_test_indices(version))
    all_doc_ids = train_doc_ids + test_doc_ids

    return train_doc_ids, test_doc_ids, all_doc_ids
