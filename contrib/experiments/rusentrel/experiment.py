import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.folding_type import FoldingType
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiments.rusentrel.utils import folding_type_to_str, get_rusentrel_inds
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
        self.__folding_type = folding_type

        super(RuSentRelExperiment, self).__init__(data_io=data_io,
                                                  experiment_io=experiment_io)

        logger.info("Create opinion oprations ... ")
        opin_ops = RuSentrelOpinionOperations(data_io=data_io,
                                              version=version,
                                              experiment_io=self.ExperimentIO)

        logger.info("Create document operations ... ")
        doc_ops = RuSentrelDocumentOperations(data_io=data_io,
                                              folding_type=folding_type,
                                              version=version)

        # Setup
        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

    @property
    def Name(self):
        return u"rsr-{version}-{format}".format(version=self.__version.value,
                                                format=folding_type_to_str(self.__folding_type))

