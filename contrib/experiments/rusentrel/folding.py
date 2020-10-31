from enum import Enum

from arekit.common.experiment.cv.base import TwoClassCVFolding
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.folding.fixed import FixedFolding
from arekit.contrib.experiments.rusentrel.utils import get_rusentrel_inds
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


class FoldingType(Enum):
    """
    Assumes a fixed separation onto train and test collections
    """
    Fixed = 1

    """
    Assumes separation using k-fold cross-validation approach
    """
    CrossValidation = 2


def create_rusentrel_experiment_data_folding(folding_type, version):
    """ Supported data folding in experiments with RuSentRelCollection.
    """
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(version, RuSentRelVersions))

    # Providing doc_ids
    train_doc_ids, test_doc_ids, all_doc_ids = get_rusentrel_inds(version)

    # We support only TRAIN and TEST subcollections
    supported_data_types = [DataType.Train, DataType.Test]

    if folding_type == FoldingType.Fixed:
        """ Fixed separation onto Train/Test experiment.
        """
        return FixedFolding(doc_to_dtype_func=None,
                            doc_ids_to_fold=all_doc_ids,
                            supported_data_types=supported_data_types)

    elif folding_type == FoldingType.CrossValidation:
        """ CV-based separation
        """
        return TwoClassCVFolding(doc_ids_to_fold=all_doc_ids,
                                 supported_data_types=supported_data_types,
                                 cv_count=3)

    raise Exception("Folding type `{}` does not supported by RuSentRel experiment".format(folding_type))