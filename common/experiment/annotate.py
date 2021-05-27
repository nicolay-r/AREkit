from arekit.common.experiment.annot.base import BaseAnnotator
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations


def do_annotation(logger, annotator, opin_ops, doc_ops):
    """ Performing annotation both using annotator and algorithm.
    """
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(annotator, BaseAnnotator))

    # Initializing annotator
    logger.info("Initializing annotator ...")
    annotator.initialize(opin_ops=opin_ops, doc_ops=doc_ops)

    # Perform annotation
    logger.info("Perform annotation ...")
    for data_type in doc_ops.DataFolding.iter_supported_data_types():
        annotator.serialize_missed_collections(data_type=data_type)
