import logging

from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.experiments.common import entity_to_group_func
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
from arekit.contrib.experiments.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)


class RuSentRelWithRuAttitudesExperiment(BaseExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Original Paper (RuAttitudes-1.0): https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, exp_data, experiment_io_type, folding_type, ruattitudes_version,
                 rusentrel_version, load_docs, extra_name_suffix):
        assert(isinstance(ruattitudes_version, RuAttitudesVersions))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(issubclass(experiment_io_type, BaseIOUtils))

        self.__ruattitudes_version = ruattitudes_version
        self.__rusentrel_version = rusentrel_version
        self.__rusentrel_synonyms = None

        logger.info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        # RuSentRel doc operations init.
        rusentrel_folding = create_rusentrel_experiment_data_folding(
            folding_type=folding_type,
            version=rusentrel_version,
            docs_reader_func=lambda doc_id: doc_ops.read_news(doc_id),
            experiment_io=experiment_io)

        # init documents.
        rusentrel_doc = RuSentrelDocumentOperations(exp_data=exp_data,
                                                    version=rusentrel_version,
                                                    folding=rusentrel_folding,
                                                    get_synonyms_func=self._get_or_load_synonyms_collection)

        # Loading ru_attitudes in memory
        ru_attitudes = read_ruattitudes_in_memory(version=ruattitudes_version,
                                                  used_doc_ids_set=set(rusentrel_doc.DataFolding.iter_doc_ids()),
                                                  keep_doc_ids_only=not load_docs)

        # RuAttitudes doc operations init.
        ruattiudes_folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.iterkeys()))
        ruattitudes_doc = RuAttitudesDocumentOperations(exp_data=exp_data,
                                                        folding=ruattiudes_folding,
                                                        ru_attitudes=ru_attitudes)

        # Init opinions
        rusentrel_op = RuSentrelOpinionOperations(experiment_data=exp_data,
                                                  version=rusentrel_version,
                                                  experiment_io=experiment_io,
                                                  get_synonyms_func=self._get_or_load_synonyms_collection)

        ruattitudes_op = RuAttitudesOpinionOperations(ru_attitudes=ru_attitudes)

        # Init experiment doc_ops and opin_ops
        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(rusentrel_doc=rusentrel_doc,
                                                             ruattitudes_doc=ruattitudes_doc)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            rusentrel_op=rusentrel_op,
            ruattitudes_op=ruattitudes_op,
            is_rusentrel_doc=lambda doc_id: rusentrel_doc.DataFolding.contains_doc_id(doc_id))

        exp_name = u"rsr-{rsr_version}-ra-{ra_version}-{folding_type}".format(
            rsr_version=self.__rusentrel_version.value,
            ra_version=self.__ruattitudes_version.value,
            folding_type=doc_ops.DataFolding.Name)

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(exp_data=exp_data,
                                                                 doc_ops=doc_ops,
                                                                 opin_ops=opin_ops,
                                                                 experiment_io=experiment_io,
                                                                 name=exp_name,
                                                                 extra_name_suffix=extra_name_suffix)

    def _get_or_load_synonyms_collection(self):
        if self.__rusentrel_synonyms is None:
            logger.info("Read synonyms collection [RuSentRel]...")
            self.__rusentrel_synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
                stemmer=self.DataIO.Stemmer,
                version=self.__rusentrel_version)

        return self.__rusentrel_synonyms

    def entity_to_group(self, entity):
        return entity_to_group_func(entity, synonyms=self.__rusentrel_synonyms)
