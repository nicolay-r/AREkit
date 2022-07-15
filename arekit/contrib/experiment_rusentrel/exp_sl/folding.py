from collections import OrderedDict

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.fixed import FixedFolding
from arekit.common.folding.types import FoldingType
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils
from arekit.contrib.utils.cv.splitters.default import SimpleCrossValidataionSplitter
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding

DEFAULT_CV_COUNT = 3


def create_rusentrel_experiment_data_folding(folding_type, version):
    """ Supported data folding in experiments with RuSentRelCollection.
    """
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(version, RuSentRelVersions))

    # Providing doc_ids
    train_doc_ids, test_doc_ids, all_doc_ids = __get_rusentrel_inds(version)

    # We support only TRAIN and TEST subcollections
    data_types = OrderedDict()
    data_types[DataType.Train] = train_doc_ids
    data_types[DataType.Test] = test_doc_ids

    supported_data_types = list(data_types.keys())

    if folding_type == FoldingType.Fixed:
        """ Fixed separation onto Train/Test experiment.
        """

        doc_to_dtype = {}
        for dtype, doc_ids in data_types.items():
            for doc_id in doc_ids:
                doc_to_dtype[doc_id] = dtype

        return FixedFolding(doc_to_datatypes_func=lambda doc_id: doc_to_dtype[doc_id],
                            doc_ids_to_fold=all_doc_ids,
                            supported_data_types=supported_data_types)

    elif folding_type == FoldingType.CrossValidation:
        """ CV-based separation
        """

        splitter = SimpleCrossValidataionSplitter()

        return TwoClassCVFolding(doc_ids_to_fold=all_doc_ids,
                                 supported_data_types=supported_data_types,
                                 cv_count=DEFAULT_CV_COUNT,
                                 splitter=splitter)

    raise Exception("Folding type `{}` does not supported by RuSentRel experiment".format(folding_type))


def __get_rusentrel_inds(version):
    """ Provides all news_inds utilized in RuSentRel collection
    """
    train_doc_ids = list(RuSentRelIOUtils.iter_train_indices(version))
    test_doc_ids = list(RuSentRelIOUtils.iter_test_indices(version))
    all_doc_ids = train_doc_ids + test_doc_ids

    return train_doc_ids, test_doc_ids, all_doc_ids
