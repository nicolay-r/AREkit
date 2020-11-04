from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator


def perform_neutral_annotation(logger, neutral_annotator, opin_ops, doc_ops):
    """ Performing annotation both using annotator and algorithm.
    """
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(neutral_annotator, BaseNeutralAnnotator))

    # Initializing annotator
    logger.info("Initializing neutral annotator ...")
    neutral_annotator.initialize(opin_ops=opin_ops, doc_ops=doc_ops)

    # Perform neutral annotation
    logger.info("Perform neutral annotation ...")
    for data_type in doc_ops.DataFolding.iter_supported_data_types():
        neutral_annotator.serialize_missed_collections(data_type=data_type)
