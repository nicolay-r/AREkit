import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiments.folding_type import FoldingType
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection

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

    def __init__(self, data_io, experiment_io_type, version, folding_type):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(issubclass(experiment_io_type, BaseIOUtils))

        logger.info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        logger.info("Read synonyms collection ...")
        synonyms = RuSentRelSynonymsCollection.load_collection(stemmer=data_io.Stemmer,
                                                               version=version)

        logger.info("Create opinion operations ... ")
        opin_ops = RuSentrelOpinionOperations(experiment_data=data_io,
                                              version=version,
                                              experiment_io=experiment_io,
                                              synonyms=synonyms)

        logger.info("Create document operations ... ")
        folding = create_rusentrel_experiment_data_folding(folding_type=folding_type,
                                                           version=version,
                                                           docs_reader_func=lambda doc_id: doc_ops.read_news(doc_id),
                                                           experiment_io=experiment_io)
        doc_ops = RuSentrelDocumentOperations(experiment_io=data_io,
                                              folding=folding,
                                              version=version,
                                              get_synonyms_func=lambda: synonyms)

        super(RuSentRelExperiment, self).__init__(data_io=data_io,
                                                  experiment_io=experiment_io,
                                                  doc_ops=doc_ops,
                                                  opin_ops=opin_ops)

        # Setup experiment name.
        self.__name = u"rsr-{version}-{format}".format(version=version.value,
                                                       format=doc_ops.DataFolding.Name)

    @property
    def Name(self):
        return self.__name

