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

    def __init__(self, exp_data, experiment_io_type, version, load_docs, extra_name_suffix):
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(load_docs, bool))

        self.__version = version
        self.__extra_name_suffix = extra_name_suffix

        logger.info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        logger.info("Loading RuAttitudes collection optionally [{version}] ...".format(version=version))
        ru_attitudes = read_ruattitudes_in_memory(version=version,
                                                  used_doc_ids_set=None,
                                                  keep_doc_ids_only=not load_docs)

        logger.info("Read synonyms collection ...")
        synonyms = RuAttitudesSynonymsCollection.load_collection(stemmer=exp_data.Stemmer,
                                                                 version=version)

        folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.iterkeys()))

        logger.info("Create document operations ... ")
        doc_ops = RuAttitudesDocumentOperations(exp_data=exp_data,
                                                folding=folding,
                                                ru_attitudes=ru_attitudes)

        logger.info("Create opinion operations ... ")
        opin_ops = RuAttitudesOpinionOperations(synonyms=synonyms,
                                                ru_attitudes=ru_attitudes)

        exp_name = u"ra-{ra_version}".format(ra_version=self.__version.value)

        super(RuAttitudesExperiment, self).__init__(exp_data=exp_data,
                                                    experiment_io=experiment_io,
                                                    opin_ops=opin_ops,
                                                    doc_ops=doc_ops,
                                                    name=exp_name,
                                                    extra_name_suffix=extra_name_suffix)
