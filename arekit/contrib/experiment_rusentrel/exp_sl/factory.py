import logging

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel import common
from arekit.contrib.experiment_rusentrel.common import create_text_parser
from arekit.contrib.experiment_rusentrel.exp_sl.documents import RuSentrelDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_sl.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_rusentrel_experiment(exp_ctx, exp_io, version, folding_type):
    """
    Represents a cv-based experiment over RuSentRel collection,
    which supports train/test separation.
    utilized in papers:
        https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10
        https://wwww.easychair.org/publications/download/pQrC
    """
    assert(isinstance(version, RuSentRelVersions))
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(exp_io, BaseIOUtils))

    synonyms_provider = OptionalSynonymsProvider(version)

    logger.info("Create opinion operations ... ")
    opin_ops = RuSentrelOpinionOperations(exp_ctx=exp_ctx,
                                          version=version,
                                          exp_io=exp_io,
                                          get_synonyms_func=synonyms_provider.get_or_load_synonyms_collection)

    text_parser = create_text_parser(
        exp_ctx=exp_ctx,
        entities_parser=RuSentRelTextEntitiesParser(),
        value_to_group_id_func=synonyms_provider.get_or_load_synonyms_collection().get_synonym_group_index)

    doc_ops = RuSentrelDocumentOperations(exp_ctx=exp_ctx,
                                          version=version,
                                          text_parser=text_parser,
                                          get_synonyms_func=synonyms_provider.get_or_load_synonyms_collection)

    return BaseExperiment(exp_ctx=exp_ctx, exp_io=exp_io, doc_ops=doc_ops, opin_ops=opin_ops)


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
