import logging

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.common import entity_to_group_func
from arekit.contrib.experiment_rusentrel.exp_ds.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_ds.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiment_rusentrel.exp_ds.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.experiment_rusentrel.exp_joined.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_joined.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_sl.documents import RuSentrelDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_sl.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiment_rusentrel.exp_sl.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelWithRuAttitudesExperiment(BaseExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Original Paper (RuAttitudes-1.0): https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, exp_data, experiment_io_type, folding_type, ruattitudes_version,
                 rusentrel_version, load_docs, extra_name_suffix, do_log=True):
        assert(isinstance(ruattitudes_version, RuAttitudesVersions))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(issubclass(experiment_io_type, BaseIOUtils))
        assert(isinstance(do_log, bool))

        # Setup logging option.
        self._init_log_flag(do_log)

        self.__ruattitudes_version = ruattitudes_version
        self.__rusentrel_version = rusentrel_version
        self.__load_docs = load_docs
        self.__exp_data = exp_data
        self.__do_log = do_log

        # To be initialized later (on demand)
        self.__rusentrel_synonyms = None
        self.__ru_attitudes = None
        self.__ruattitudes_doc = None
        self.__ruattitudes_op = None

        self.log_info("Init experiment io ...")
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
        self.__rusentrel_doc_ids = rusentrel_doc.DataFolding.iter_doc_ids()

        # Init opinions
        rusentrel_op = RuSentrelOpinionOperations(experiment_data=exp_data,
                                                  version=rusentrel_version,
                                                  experiment_io=experiment_io,
                                                  get_synonyms_func=self._get_or_load_synonyms_collection)

        # Init experiment doc_ops and opin_ops
        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(
            rusentrel_doc=rusentrel_doc,
            get_ruattitudes_doc=self.__get_or_load_ruattitudes_doc_ops)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            rusentrel_op=rusentrel_op,
            get_ruattitudes_op=self.__get_or_load_ruattitudes_opin_ops,
            is_rusentrel_doc=lambda doc_id: rusentrel_doc.DataFolding.contains_doc_id(doc_id))

        exp_name = "rsr-{rsr_version}-ra-{ra_version}-{folding_type}".format(
            rsr_version=self.__rusentrel_version.value,
            ra_version=self.__ruattitudes_version.value,
            folding_type=doc_ops.DataFolding.Name)

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(exp_data=exp_data,
                                                                 doc_ops=doc_ops,
                                                                 opin_ops=opin_ops,
                                                                 experiment_io=experiment_io,
                                                                 name=exp_name,
                                                                 extra_name_suffix=extra_name_suffix)

    # region private methods

    def __get_or_load_ruattitudes_opin_ops(self):
        if self.__ruattitudes_op is None:
            if self.__ru_attitudes is None:
                self.__init_ruattitudes_and_doc_ops()
            self.__ruattitudes_op = RuAttitudesOpinionOperations(ru_attitudes=self.__ru_attitudes)
        return self.__ruattitudes_op

    def __get_or_load_ruattitudes_doc_ops(self):
        if self.__ruattitudes_doc is None:
            self.__init_ruattitudes_and_doc_ops()
        return self.__ruattitudes_doc

    def __init_ruattitudes_and_doc_ops(self):
        """ this operation may take a while since it assumes to
            load ru_attitudes collection in memory.
        """
        # Loading ru_attitudes in memory
        ru_attitudes = read_ruattitudes_in_memory(
            version=self.__ruattitudes_version,
            used_doc_ids_set=set(self.__rusentrel_doc_ids),
            keep_doc_ids_only=not self.__load_docs)

        # RuAttitudes doc operations init.
        ruattiudes_folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.keys()))

        # Completing initialization.
        self.__ruattitudes_doc = RuAttitudesDocumentOperations(
            exp_data=self.__exp_data,
            folding=ruattiudes_folding,
            ru_attitudes=ru_attitudes)
        self.__ru_attitudes = ru_attitudes

    # endregion

    def _get_or_load_synonyms_collection(self):
        if self.__rusentrel_synonyms is None:
            self.log_info("Read synonyms collection [RuSentRel]...")
            self.__rusentrel_synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
                stemmer=self.DataIO.Stemmer,
                version=self.__rusentrel_version)

        return self.__rusentrel_synonyms

    def entity_to_group(self, entity):
        return entity_to_group_func(entity, synonyms=self.__rusentrel_synonyms)
