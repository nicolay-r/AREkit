from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


class RuSentRelWithRuAttitudesExperiment(CVBasedExperiment):
    """ Application of distant supervision, especially for pretraining purposes
    """

    def __init__(self, data_io, prepare_model_root, used_doc_ids_set, version, ra_instance=None):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(used_doc_ids_set, set))
        assert(isinstance(ra_instance, dict) or ra_instance is None)

        self.__version = version

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(
            data_io=data_io,
            prepare_model_root=prepare_model_root)

        doc_ops = RuAttitudesDocumentOperations(data_io=data_io)

        neutral_root = get_path_of_subfolder_in_experiments_dir(
            experiments_dir=data_io.get_input_samples_dir(self.Name),
            subfolder_name=self.get_annot_name())

        opin_ops = RuAttitudesOpinionOperations(synonyms=data_io.SynonymsCollection,
                                                neutral_root=neutral_root)

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


