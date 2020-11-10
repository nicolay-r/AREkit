import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.ruattitudes.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.synonyms import RuAttitudesSynonymsCollection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuAttitudesExperiment(BaseExperiment):
    """ Application of distant supervision, especially for pretraining purposes.
        Suggested to utilize with a large RuAttitudes-format collections (v2.0-large).
    """

    def __init__(self, data_io, experiment_io_type, version, load_ruatittudes):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(load_ruatittudes, bool))

        self.__version = version

        logger.info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        logger.info("Loading RuAttitudes collection optionally [{version}] ...".format(version=version))
        ru_attitudes = read_ruattitudes_in_memory(version=version, used_doc_ids_set=None) \
            if load_ruatittudes else None

        logger.info("Read synonyms collection ...")
        synonyms = RuAttitudesSynonymsCollection.load_collection(stemmer=data_io.Stemmer,
                                                                 version=version)

        folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.iterkeys()))

        logger.info("Create document operations ... ")
        doc_ops = RuAttitudesDocumentOperations(data_io=data_io,
                                                folding=folding,
                                                ru_attitudes=ru_attitudes)

        logger.info("Create opinion operations ... ")
        opin_ops = RuAttitudesOpinionOperations(synonyms=synonyms,
                                                ru_attitudes=ru_attitudes)

        super(RuAttitudesExperiment, self).__init__(data_io=data_io,
                                                    experiment_io=experiment_io,
                                                    opin_ops=opin_ops,
                                                    doc_ops=doc_ops)

    @property
    def Name(self):
        return u"ra-{ra_version}".format(ra_version=self.__version.value)


