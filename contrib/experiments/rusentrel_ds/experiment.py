import logging

from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.ruattitudes.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiments.rusentrel_ds.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel_ds.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.synonyms import RuAttitudesSynonymsCollection
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection

logger = logging.getLogger(__name__)


class RuSentRelWithRuAttitudesExperiment(BaseExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Original Paper (RuAttitudes-1.0): https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, data_io, experiment_io_type, folding_type,
                 ruattitudes_version, rusentrel_version, ra_instance=None):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(ruattitudes_version, RuAttitudesVersions))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_instance, dict) or ra_instance is None)
        assert(issubclass(experiment_io_type, BaseIOUtils))

        logger.info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        self.__ruattitudes_version = ruattitudes_version
        self.__rusentrel_version = rusentrel_version

        logger.info("Read synonyms collection [RuSentRel]...")
        rusentrel_synonyms = RuSentRelSynonymsCollection.load_collection(stemmer=data_io.Stemmer,
                                                                         version=rusentrel_version)

        logger.info("Read synonyms collection [RuAttitudes]...")
        ruattitudes_synonyms = RuAttitudesSynonymsCollection.load_collection(stemmer=data_io.Stemmer,
                                                                             version=ruattitudes_version)

        logger.info("Merging collections [RuSentRel <- RuAttitudes]...")
        joined_synonyms = None

        # RuSentRel doc operations init.
        rusentrel_folding = create_rusentrel_experiment_data_folding(
            folding_type=folding_type,
            version=rusentrel_version,
            docs_reader_func=lambda doc_id: doc_ops.read_news(doc_id),
            experiment_io=experiment_io)

        # init documents.
        rusentrel_doc = RuSentrelDocumentOperations(experiment_io=data_io,
                                                    version=rusentrel_version,
                                                    folding=rusentrel_folding,
                                                    get_synonyms_func=lambda: rusentrel_synonyms)

        # Loading ru_attitudes in memory
        ru_attitudes = ra_instance
        if ra_instance is None:
            ru_attitudes = read_ruattitudes_in_memory(version=ruattitudes_version,
                                                      used_doc_ids_set=set(rusentrel_doc.DataFolding.iter_doc_ids()))

        # RuAttitudes doc operations init.
        ruatttiudes_folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.iterkeys()))
        ruattitudes_doc = RuAttitudesDocumentOperations(data_io=data_io,
                                                        folding=ruatttiudes_folding)

        # Init opinions
        rusentrel_op = RuSentrelOpinionOperations(experiment_data=data_io,
                                                  version=rusentrel_version,
                                                  experiment_io=experiment_io,
                                                  synonyms=rusentrel_synonyms)

        ruattitudes_op = RuAttitudesOpinionOperations(ruattitudes_synonyms)

        # Init experiment doc_ops and opin_ops
        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(rusentrel_doc=rusentrel_doc,
                                                             ruattitudes_doc=ruattitudes_doc)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            synonyms=joined_synonyms,
            rusentrel_op=rusentrel_op,
            ruattitudes_op=ruattitudes_op,
            is_rusentrel_doc=lambda doc_id: rusentrel_doc.DataFolding.contains_doc_id(doc_id))

        # Providing RuAttitudes instance into doc and opins instances.
        ruattitudes_doc.set_ru_attitudes(ru_attitudes)
        ruattitudes_op.set_ru_attitudes(ru_attitudes)

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(data_io=data_io,
                                                                 doc_ops=doc_ops,
                                                                 opin_ops=opin_ops,
                                                                 experiment_io=experiment_io)

        # Composing experiment name
        self.__name = u"rsr-{rsr_version}-ra-{ra_version}-{folding_type}".format(
            rsr_version=self.__rusentrel_version.value,
            ra_version=self.__ruattitudes_version.value,
            folding_type=self.DocumentOperations.DataFolding.Name)

    @property
    def Name(self):
        return self.__name
