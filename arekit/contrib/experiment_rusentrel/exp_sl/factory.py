import logging

from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.experiment_rusentrel import common
from arekit.contrib.experiment_rusentrel.base import CustomExperiment
from arekit.contrib.experiment_rusentrel.exp_sl.documents import RuSentrelDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_sl.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_rusentrel_experiment(exp_ctx, data_folding, exp_io, version, result_target_dir):
    """
    Represents a cv-based experiment over RuSentRel collection,
    which supports train/test separation.
    utilized in papers:
        https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10
        https://wwww.easychair.org/publications/download/pQrC
    """
    assert(isinstance(exp_ctx, ExperimentContext))
    assert(isinstance(data_folding, BaseDataFolding))
    assert(isinstance(exp_io, BaseIOUtils))
    assert(isinstance(version, RuSentRelVersions))

    synonyms_provider = OptionalSynonymsProvider(version)

    logger.info("Create opinion operations ... ")
    opin_ops = RuSentrelOpinionOperations(data_folding=data_folding,
                                          version=version,
                                          exp_io=exp_io,
                                          labels_count=exp_ctx.LabelsCount,
                                          get_synonyms_func=synonyms_provider.get_or_load_synonyms_collection,
                                          result_target_dir=result_target_dir)

    doc_ops = RuSentrelDocumentOperations(version=version,
                                          get_synonyms_func=synonyms_provider.get_or_load_synonyms_collection)

    return CustomExperiment(exp_ctx=exp_ctx, exp_io=exp_io, doc_ops=doc_ops, opin_ops=opin_ops)


class OptionalSynonymsProvider(object):

    def __init__(self, version):
        self.__rsr_version = version
        self.__synonyms = None

    def get_or_load_synonyms_collection(self):
        if self.__synonyms is None:
            self.__synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
                stemmer=common.create_stemmer(),
                version=self.__rsr_version)

        return self.__synonyms
