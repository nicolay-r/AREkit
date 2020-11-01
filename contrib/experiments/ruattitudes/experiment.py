import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.ruattitudes.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuAttitudesExperiment(BaseExperiment):
    """ Application of distant supervision, especially for pretraining purposes.
        Suggested to utilize with a large RuAttitudes-format collections (v2.0-large).
    """

    def __init__(self, data_io, experiment_io, version, used_doc_ids_set=None, ra_instance=None):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(used_doc_ids_set, set) or used_doc_ids_set is None)
        assert(isinstance(ra_instance, dict) or ra_instance is None)

        self.__version = version

        ru_attitudes = ra_instance
        if ra_instance is None:
            ru_attitudes = read_ruattitudes_in_memory(
                version=version,
                used_doc_ids_set=used_doc_ids_set)

        super(RuAttitudesExperiment, self).__init__(data_io=data_io,
                                                    experiment_io=experiment_io)

        logger.info("Read synonyms collection ...")
        synonyms = None

        folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.iterkeys()))

        logger.info("Create document operations ... ")
        doc_ops = RuAttitudesDocumentOperations(data_io=data_io,
                                                folding=folding)

        logger.info("Create opinion operations ... ")
        opin_ops = RuAttitudesOpinionOperations(synonyms)

        doc_ops.set_ru_attitudes(ru_attitudes)
        opin_ops.set_ru_attitudes(ru_attitudes)

        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

    @property
    def Name(self):
        return u"ra-{ra_version}".format(ra_version=self.__version.value)


