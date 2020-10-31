from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils


def get_rusentrel_inds(version):
    """ Provides all news_inds utilized in RuSentRel collection
    """
    train_doc_ids = list(RuSentRelIOUtils.iter_train_indices(version))
    test_doc_ids = list(RuSentRelIOUtils.iter_test_indices(version))
    all_doc_ids = train_doc_ids + test_doc_ids

    return train_doc_ids, test_doc_ids, all_doc_ids
