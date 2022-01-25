import logging

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel import common
from arekit.contrib.experiment_rusentrel.common import create_text_parser
from arekit.contrib.experiment_rusentrel.exp_ds.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_ds.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiment_rusentrel.exp_ds.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.experiment_rusentrel.exp_joined.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_joined.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_sl.documents import RuSentrelDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_sl.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelWithRuAttitudesExperiment(BaseExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Original Paper (RuAttitudes-1.0): https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, exp_ctx, experiment_io_type, folding_type, ruattitudes_version,
                 rusentrel_version, load_docs, do_log=True):
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
        self.__exp_ctx = exp_ctx
        self.__do_log = do_log

        # To be initialized later (on demand)
        self.__rusentrel_synonyms = None
        self.__ru_attitudes = None
        self.__ruattitudes_doc = None
        self.__ruattitudes_op = None

        self.log_info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        # init text parser.
        # TODO. Limitation, depending on document, entities parser may vary.
        text_parser = create_text_parser(
            exp_ctx=self.__exp_ctx,
            entities_parser=RuSentRelTextEntitiesParser(),
            value_to_group_id_func=self._get_synonyms().get_synonym_group_index)

        # init documents.
        rusentrel_doc = RuSentrelDocumentOperations(exp_ctx=exp_ctx,
                                                    version=rusentrel_version,
                                                    get_synonyms_func=self._get_synonyms,
                                                    text_parser=text_parser)
        self.__rusentrel_doc_ids = exp_ctx.DataFolding.iter_doc_ids()

        # Init opinions
        rusentrel_op = RuSentrelOpinionOperations(exp_ctx=exp_ctx,
                                                  version=rusentrel_version,
                                                  experiment_io=experiment_io,
                                                  get_synonyms_func=self._get_synonyms)

        # Init experiment doc_ops and opin_ops
        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(
            rusentrel_doc=rusentrel_doc,
            get_ruattitudes_doc=self.__get_or_load_ruattitudes_doc_ops,
            text_parser=text_parser)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            rusentrel_op=rusentrel_op,
            get_ruattitudes_op=self.__get_or_load_ruattitudes_opin_ops,
            is_rusentrel_doc=lambda doc_id: exp_ctx.DataFolding.contains_doc_id(doc_id))

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(
            exp_ctx=exp_ctx, doc_ops=doc_ops, opin_ops=opin_ops, experiment_io=experiment_io)

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

        text_parser = create_text_parser(
            exp_ctx=self.__exp_ctx,
            entities_parser=RuAttitudesTextEntitiesParser(),
            value_to_group_id_func=self._get_synonyms().get_synonym_group_index)

        # Completing initialization.
        self.__ruattitudes_doc = RuAttitudesDocumentOperations(
            folding=ruattiudes_folding,
            ru_attitudes=ru_attitudes,
            text_parser=text_parser)
        self.__ru_attitudes = ru_attitudes

    # endregion

    def _get_synonyms(self):
        if self.__rusentrel_synonyms is None:
            self.log_info("Read synonyms collection [RuSentRel]...")
            self.__rusentrel_synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
                stemmer=common.create_stemmer(),
                version=self.__rusentrel_version)

        return self.__rusentrel_synonyms
