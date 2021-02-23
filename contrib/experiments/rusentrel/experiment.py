import logging

from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.experiments.common import entity_to_group_func
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiments.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelExperiment(BaseExperiment):
    """
    Represents a cv-based experiment over RuSentRel collection,
    which supports train/test separation.
    utilized in papers:
        https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10
        https://wwww.easychair.org/publications/download/pQrC
    """

    def __init__(self, exp_data, experiment_io_type, version, folding_type, extra_name_suffix):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(issubclass(experiment_io_type, BaseIOUtils))

        self.__rsr_version = version
        self.__synonyms = None

        logger.info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        logger.info("Create opinion operations ... ")
        opin_ops = RuSentrelOpinionOperations(experiment_data=exp_data,
                                              version=version,
                                              experiment_io=experiment_io,
                                              get_synonyms_func=self._get_or_load_synonyms_collection)

        logger.info("Create document operations ... ")
        folding = create_rusentrel_experiment_data_folding(folding_type=folding_type,
                                                           version=version,
                                                           docs_reader_func=lambda doc_id: doc_ops.read_news(doc_id),
                                                           experiment_io=experiment_io)
        doc_ops = RuSentrelDocumentOperations(exp_data=exp_data,
                                              folding=folding,
                                              version=version,
                                              get_synonyms_func=self._get_or_load_synonyms_collection)

        exp_name = u"rsr-{version}-{format}".format(version=version.value,
                                                    format=doc_ops.DataFolding.Name)

        super(RuSentRelExperiment, self).__init__(exp_data=exp_data,
                                                  experiment_io=experiment_io,
                                                  doc_ops=doc_ops,
                                                  opin_ops=opin_ops,
                                                  name=exp_name,
                                                  extra_name_suffix=extra_name_suffix)

    def _get_or_load_synonyms_collection(self):
        if self.__synonyms is None:
            logger.info("Read synonyms collection ...")
            self.__synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
                stemmer=self.DataIO.Stemmer,
                version=self.__rsr_version)

        return self.__synonyms

    def entity_to_group(self, entity):
        return entity_to_group_func(entity, synonyms=self.__synonyms)
