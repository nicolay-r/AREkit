from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.contrib.experiments.common import get_neutral_annotation_root
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


class RuAttitudesExperiment(CVBasedExperiment):
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

        super(RuAttitudesExperiment, self).__init__(data_io=data_io,
                                                    experiment_io=experiment_io)

        doc_ops = RuAttitudesDocumentOperations(data_io=data_io)

        opin_ops = RuAttitudesOpinionOperations(synonyms=data_io.SynonymsCollection,
                                                neutral_root=get_neutral_annotation_root(self))

        ru_attitudes = ra_instance
        if ra_instance is None:
            ru_attitudes = read_ruattitudes_in_memory(
                version=version,
                used_doc_ids_set=used_doc_ids_set)

        doc_ops.set_ru_attitudes(ru_attitudes)
        opin_ops.set_ru_attitudes(ru_attitudes)

        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

    @property
    def Name(self):
        return u"ra-{ra_version}".format(ra_version=self.__version.value)


