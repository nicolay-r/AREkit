import logging

from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.contrib.experiment_rusentrel.base import BaseExperiment
from arekit.contrib.experiment_rusentrel.exp_ds.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_ds.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_ruattitudes_experiment(exp_ctx, exp_io, version, load_docs, ra_doc_ids_func):
    """ Application of distant supervision, especially for pretraining purposes.
        Suggested to utilize with a large RuAttitudes-format collections (v2.0-large).
    """
    assert(isinstance(version, RuAttitudesVersions))
    assert(isinstance(exp_io, BaseIOUtils))
    assert(isinstance(load_docs, bool))

    ru_attitudes = read_ruattitudes_in_memory(version=version,
                                              doc_id_func=ra_doc_ids_func,
                                              keep_doc_ids_only=not load_docs)

    logger.info("Create document operations ...")
    doc_ops = RuAttitudesDocumentOperations(ru_attitudes=ru_attitudes)

    logger.info("Create opinion operations ...")
    opin_ops = RuAttitudesOpinionOperations(ru_attitudes=ru_attitudes)

    return BaseExperiment(exp_ctx=exp_ctx, exp_io=exp_io, opin_ops=opin_ops, doc_ops=doc_ops)
