import logging

from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelExperiment(CVBasedExperiment):
    """
    Represents a cv-based experiment over RuSentRel collection,
    which supports train/test separation.
    utilized in papers:
        https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10
        https://wwww.easychair.org/publications/download/pQrC
    """

    def __init__(self, data_io, version):
        assert(isinstance(version, RuSentRelVersions))

        self.__version = version

        super(RuSentRelExperiment, self).__init__(data_io=data_io)

        neutral_root = get_path_of_subfolder_in_experiments_dir(
            experiments_dir=data_io.get_input_samples_dir(self.Name),
            subfolder_name=self.get_annot_name())

        logger.info("Create opinion oprations ... ")
        opin_ops = RuSentrelOpinionOperations(data_io=data_io,
                                              version=version,
                                              neutral_root=neutral_root,
                                              rusentrel_news_ids=self.get_rusentrel_inds())

        logger.info("Create document operations ... ")
        doc_ops = RuSentrelDocumentOperations(data_io=data_io,
                                              rusentrel_version=version)

        # Setup
        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

    @property
    def Name(self):
        return u"rusentrel-{version}".format(version=self.__version.value)

    @staticmethod
    def get_rusentrel_inds():
        return set(RuSentRelIOUtils.iter_collection_indices())

