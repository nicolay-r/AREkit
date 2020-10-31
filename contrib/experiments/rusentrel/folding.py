from collections import OrderedDict
from os.path import join

from enum import Enum

from arekit.common.experiment.cv.base import TwoClassCVFolding
from arekit.common.experiment.cv.doc_stat.sentence import SentenceBasedDocumentStatGenerator
from arekit.common.experiment.cv.splitters.statistical import StatBasedCrossValidataionSplitter
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.folding.fixed import FixedFolding
from arekit.common.experiment.io_utils import BaseIOUtils
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


def create_rusentrel_experiment_data_folding(folding_type, version, docs_reader_func, experiment_io):
    """ Supported data folding in experiments with RuSentRelCollection.
    """
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(version, RuSentRelVersions))
    assert(callable(docs_reader_func))
    assert(isinstance(experiment_io, BaseIOUtils))

    # Providing doc_ids
    train_doc_ids, test_doc_ids, all_doc_ids = get_rusentrel_inds(version)

    # We support only TRAIN and TEST subcollections
    data_types = OrderedDict()
    data_types[DataType.Train] = train_doc_ids
    data_types[DataType.Test] = test_doc_ids

    supported_data_types = list(data_types.keys())

    if folding_type == FoldingType.Fixed:
        """ Fixed separation onto Train/Test experiment.
        """

        doc_to_dtype = {}
        for dtype, doc_ids in data_types.iteritems():
            for doc_id in doc_ids:
                doc_to_dtype[doc_id] = dtype

        return FixedFolding(doc_to_dtype_func=lambda doc_id: doc_to_dtype[doc_id],
                            doc_ids_to_fold=all_doc_ids,
                            supported_data_types=supported_data_types)

    elif folding_type == FoldingType.CrossValidation:
        """ CV-based separation
        """

        # We utilize sentence-based cv-splitter.
        splitter = StatBasedCrossValidataionSplitter(
            docs_stat=SentenceBasedDocumentStatGenerator(docs_reader_func),
            docs_stat_filepath_func=lambda: join(experiment_io.get_target_dir(), u"docs_stat.txt"))

        return TwoClassCVFolding(doc_ids_to_fold=all_doc_ids,
                                 supported_data_types=supported_data_types,
                                 cv_count=3,
                                 splitter=splitter)

    raise Exception("Folding type `{}` does not supported by RuSentRel experiment".format(folding_type))