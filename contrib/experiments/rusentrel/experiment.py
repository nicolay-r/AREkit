import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiments.folding_type import FoldingType
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
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

    def __init__(self, data_io, experiment_io, version, folding_type):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))

        self.__version = version

        super(RuSentRelExperiment, self).__init__(data_io=data_io,
                                                  experiment_io=experiment_io)

        logger.info("Create opinion oprations ... ")
        opin_ops = RuSentrelOpinionOperations(data_io=data_io,
                                              version=version,
                                              experiment_io=self.ExperimentIO)

        logger.info("Create document operations ... ")
        folding = create_rusentrel_experiment_data_folding(folding_type=folding_type,
                                                           version=version,
                                                           docs_reader_func=lambda doc_id: doc_ops.read_news(doc_id),
                                                           experiment_io=self.ExperimentIO)
        doc_ops = RuSentrelDocumentOperations(data_io=data_io,
                                              folding=folding,
                                              version=version)

        # Setup
        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

    @property
    def Name(self):
        return u"rsr-{version}-{format}".format(version=self.__version.value,
                                                format=self.DocumentOperations.DataFolding.Name)

